import time
from picamera import PiCamera
from sense_hat import SenseHat
from datetime import datetime
import cv2
import math
import os

# Initialize camera and Sense HAT
camera = PiCamera()
sense = SenseHat()

# Set camera resolution if needed
camera.resolution = (1024, 768)

# Function to capture and process images
def capture_and_process(image_counter):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    image_path = f'/home/pi/images/image_{image_counter}_{timestamp}.jpg'
    camera.capture(image_path)
    sense.show_message(f"Captured {image_counter}", scroll_speed=0.05)
    return image_path

# Function to process two images and calculate speed
def process_images(image_1, image_2):
    try:
        time_difference = get_time_difference(image_1, image_2)
        image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
        keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
        matches = calculate_matches(descriptors_1, descriptors_2)
        coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
        average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
        speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
        sense.show_message(f"Speed: {speed:.2f} km/s", scroll_speed=0.05)
        # Removed OpenCV display function to prevent window from opening
    except Exception as e:
        sense.show_message(f"Error: {e}", scroll_speed=0.05)

# Main loop to run for 4 minutes
start_time = time.time()
image_counter = 0
previous_image = None
duration = 4 * 60  # 4 minutes in seconds

# Open the result.txt file in write mode
with open('result.txt', 'w') as f:
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break
        current_image = capture_and_process(image_counter)
        if previous_image:
            process_images(previous_image, current_image)
        previous_image = current_image
        image_counter += 1
        time.sleep(10)  # Adjust the interval as needed

    # Write a completion message to the file
    f.write("Image processing completed successfully.\n")

# Cleanup
camera.close()
sense.clear()

# Existing functions from your script
def get_time(image):
    # Open image file
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        # Check if EXIF metadata is available
        if not img.has_exif:
            raise ValueError(f"Image {image} does not contain EXIF metadata.")
        
        # Get datetime_original field
        time_str = img.get("datetime_original")
        if not time_str:
            raise ValueError(f"Image {image} does not contain 'datetime_original' in EXIF metadata.")
        
        # Parse the time string into a datetime object
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time

def get_time_difference(image_1, image_2):
    # Calculate time difference between two images
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

def convert_to_cv(image_1, image_2):
    # Load images using OpenCV
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    if image_1_cv is None or image_2_cv is None:
        raise FileNotFoundError("One or both image files could not be loaded. Check the file paths.")
    return image_1_cv, image_2_cv

def calculate_features(image_1_cv, image_2_cv, feature_number):
    # Detect and compute ORB features
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    # Match descriptors using Brute Force Matcher
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    # Draw matches and display them
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600, 600), interpolation=cv2.INTER_AREA)
    # Removed OpenCV display function to prevent window from opening
    # cv2.imshow('matches', resize)
    # cv2.waitKey(0)
    # cv2.destroyWindow('matches')
