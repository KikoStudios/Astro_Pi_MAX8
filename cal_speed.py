from exif import Image
from datetime import datetime
import cv2
import math
import os


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
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    # Extract matching keypoints
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    # Calculate the mean distance between matching coordinates
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances += distance
    return all_distances / len(merged_coordinates)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    # Convert feature distance to kilometers per second
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed


# Main script
if __name__ == "__main__":
    image_1 = 'Obrazky/photo_0675.jpg'
    image_2 = 'Obrazky/photo_0676.jpg'

    if not os.path.exists(image_1) or not os.path.exists(image_2):
        print(f"Error: Ensure both {image_1} and {image_2} exist in the directory.")
    else:
        try:
            # Process images and calculate speed
            time_difference = get_time_difference(image_1, image_2)
            image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
            matches = calculate_matches(descriptors_1, descriptors_2)
            coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
            average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
            speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
            print(f"Speed: {speed} km/s")
            display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)
        except Exception as e:
            print(f"An error occurred: {e}")
#SKIBIDI MEAW