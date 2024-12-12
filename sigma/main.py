import os
import time
from datetime import datetime
import cv2
import math
from exif import Image
from picamera import PiCamera


def get_time(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            img = Image(image_file)
            time_str = img.get("datetime_original", None)
            if time_str:
                return datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    except Exception:
        return None


def capture_images(camera, num_images=2, interval=1):
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')

    image_paths = []
    for i in range(num_images):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'captured_images/image_{timestamp}.jpg'
        camera.capture(filename)
        image_paths.append(filename)
        time.sleep(interval)

    return image_paths


def process_images(image_1_path, image_2_path, gsd=12648):
    time_difference = (get_time(image_2_path) - get_time(image_1_path)).total_seconds()

    image_1_cv = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)
    image_2_cv = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(nfeatures=1000)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)

    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    coordinates_1 = [keypoints_1[match.queryIdx].pt for match in matches]
    coordinates_2 = [keypoints_2[match.trainIdx].pt for match in matches]

    distances = [math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                 for c1, c2 in zip(coordinates_1, coordinates_2)]

    average_feature_distance = sum(distances) / len(distances) if distances else 0

    distance = average_feature_distance * gsd / 100000
    speed = distance / time_difference

    return speed


def main():
    camera = PiCamera()
    camera.resolution = (2592, 1944)
    camera.rotation = 180

    image_paths = capture_images(camera, num_images=2, interval=5)

    speed = process_images(image_paths[0], image_paths[1])

    with open('result.txt', 'w') as f:
        f.write(f"{speed:.4f}")

    camera.close()


if __name__ == "__main__":
    main()