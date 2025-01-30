import cv2
import numpy as np
import math
import time
from picamera import PiCamera
from time import sleep
import logzero
from logzero import logger

# Initialize logzero logger
logzero.logfile("mission_log.txt", maxBytes=1e6, backupCount=3)
logger.setLevel(logzero.logging.DEBUG)


def capture_images(interval=15, duration=600):
    logger.info("Starting image capture process")
    # Create an instance of the PiCamera class
    camera = PiCamera()
    camera.resolution = (1920, 1080)

    # Calculate the number of images to capture
    num_images = duration // interval
    timestamps = []
    image_files = []

    for i in range(num_images):
        filename = f'image_{i}.jpg'
        camera.capture(filename)
        timestamps.append(time.time())  # Add timestamp for each image captured
        image_files.append(filename)  # Add the filename for later processing
        logger.debug(f"Captured image {filename} at {timestamps[-1]}")
        sleep(interval)

    camera.close()  # Close the camera after capturing images
    logger.info("Image capture process completed")
    return image_files, timestamps


def calculate_gsd(sensor_width_mm=6.3, image_width_px=1920, focal_length_mm=3.04, altitude_km=400):
    logger.info(f"Calculating Ground Sampling Distance (GSD) for altitude {altitude_km} km")
    altitude_m = altitude_km * 1000  # Convert altitude to meters
    gsd = (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_px)
    logger.debug(f"GSD calculated: {gsd:.6f} meters per pixel")
    return gsd  # GSD in meters per pixel


def process_images(image_files):
    logger.info("Processing captured images for feature matching")
    # Read the images in grayscale
    images = [cv2.imread(img, 0) for img in image_files]
    orb = cv2.ORB_create(nfeatures=1000)  # Create ORB feature detector

    keypoints = []
    descriptors = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors[0], descriptors[1])
    matches = sorted(matches, key=lambda x: x.distance)

    logger.debug(f"Found {len(matches)} matches between images")
    return keypoints, matches


def calculate_speed(keypoints, matches, gsd, time_interval):
    logger.info("Calculating speed from keypoints and matches")
    # Calculate the average pixel displacement and convert to meters and speed
    displacements = []
    for match in matches:
        pt1 = keypoints[0][match.queryIdx].pt
        pt2 = keypoints[1][match.trainIdx].pt
        pixel_distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        displacements.append(pixel_distance)

    if not displacements:
        logger.warning("No valid keypoint matches found, cannot calculate speed")
        return 0

    avg_pixel_displacement = np.mean(displacements)
    distance_m = avg_pixel_displacement * gsd  # Convert pixel distance to meters
    speed_kmps = (distance_m / 1000) / time_interval  # Convert meters to kilometers per second
    logger.debug(f"Calculated speed: {speed_kmps:.6f} km/s")
    return speed_kmps

#ddd
def main():
    logger.info("Starting the speed calculation process")
    image_files, timestamps = capture_images()  # Capture images and get timestamps
    time_interval = timestamps[1] - timestamps[0]  # Calculate time between captures
    logger.debug(f"Time interval between captures: {time_interval:.6f} seconds")

    gsd = calculate_gsd()  # Calculate Ground Sampling Distance
    keypoints, matches = process_images(image_files)  # Process the images to find matches
    speed = calculate_speed(keypoints, matches, gsd, time_interval)  # Calculate speed

    # Saving the result to a file
    with open('result.txt', 'w') as f:
        f.write(f'{speed:.2f}')  # Save the result to a file
    logger.info(f"Speed calculated and saved: {speed:.2f} km/s")

    print(f'Speed: {speed:.2f} km/s')


if __name__ == '__main__':
    main()
