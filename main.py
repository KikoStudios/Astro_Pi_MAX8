import os
from datetime import datetime
from picamera import PiCamera
import cv2
import math
from exif import Image


def get_time(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            img = Image(image_file)
            time_str = img.get("datetime_original", None)
            if time_str:
                return datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"Error getting time from {image_path}: {e}")
    return None


def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)

    if time_1 and time_2:
        time_difference = time_2 - time_1
        return time_difference.total_seconds()
    return None


def convert_to_cv(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def calculate_features(image, feature_number=1000):
    orb = cv2.ORB_create(nfeatures=feature_number)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    return sorted(matches, key=lambda x: x.distance)


def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        (x1, y1) = keypoints_1[match.queryIdx].pt
        (x2, y2) = keypoints_2[match.trainIdx].pt
        coordinates_1.append((x1, y1))
        coordinates_2.append((x2, y2))
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    distances = [math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                 for c1, c2 in zip(coordinates_1, coordinates_2)]
    return sum(distances) / len(distances) if distances else 0


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    if time_difference == 0:
        return 0

    distance = feature_distance * GSD / 100000
    speed = distance / time_difference * 3600
    return speed


def process_images(image_1_path, image_2_path, gsd=12648):
    time_difference = get_time_difference(image_1_path, image_2_path)
    if time_difference is None:
        return None

    image_1_cv = convert_to_cv(image_1_path)
    image_2_cv = convert_to_cv(image_2_path)

    if image_1_cv is None or image_2_cv is None:
        return None

    keypoints_1, descriptors_1 = calculate_features(image_1_cv)
    keypoints_2, descriptors_2 = calculate_features(image_2_cv)

    matches = calculate_matches(descriptors_1, descriptors_2)

    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)

    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    speed = calculate_speed_in_kmps(average_feature_distance, gsd, time_difference)

    return {
        'time_difference': time_difference,
        'average_feature_distance': average_feature_distance,
        'speed_kmph': speed
    }


def main():
    images_dir = '/path/to/images/'  # Update with your image directory
    results_path = '/path/to/results.txt'  # Update with your results file path

    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])

    with open(results_path, 'w') as results_file:
        for i in range(len(image_files) - 1):
            image_1 = image_files[i]
            image_2 = image_files[i + 1]

            result = process_images(image_1, image_2)
            if result:
                results_file.write(f"Analysis between {os.path.basename(image_1)} and {os.path.basename(image_2)}:\n")
                results_file.write(f"Time Difference: {result['time_difference']} seconds\n")
                results_file.write(f"Average Feature Distance: {result['average_feature_distance']:.2f} pixels\n")
                results_file.write(f"Estimated Speed: {result['speed_kmph']:.2f} km/h\n")
                results_file.write("---\n")

                print(f"Processed {os.path.basename(image_1)} and {os.path.basename(image_2)}")


if __name__ == "__main__":
    main()