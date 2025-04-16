import cv2
import numpy as np
import os
from setup.config import login_statement, user_folder_config
from classification.feature_extraction import calculate_contour_properties
import pandas as pd

def calculate_average_features(USER_FOLDER_PATH, image_name):
    """
    Calculate the average features for each class of images in a specified directory.

    Args:
    - USER_FOLDER_PATH: The base directory path where the images and CSV file are stored.
    - image_name: The name of the image to process.

    Returns:
    - df: A DataFrame containing the average features for each class.
    """

    # Initialize class features
    class_features = {'rbc': [], 'wbc': [], 'plt': [], 'agg': [], 'oof': []}

    # Define the directory where the images are stored
    directory = f'{USER_FOLDER_PATH}/prediction/{image_name}/boxes'

    # Define the path to the CSV file that contains the labels
    csv_file_path = f'{USER_FOLDER_PATH}/prediction/{image_name}/{image_name}.csv'

    # Load the labels from the CSV file
    labels = pd.read_csv(csv_file_path)

    # Get a list of all .png files in the directory
    png_files = [f for f in os.listdir(directory) if f.endswith('.png')]

    # Sort the list by the integer value of the filename
    png_files_sorted = sorted(png_files, key=lambda f: int(f.replace('box_', '').replace('.png', '')))

    # loop over the sorted list
    for filename in png_files_sorted:
        # Only process .png files
        if filename.endswith('.png'):
            # Extract the number from the filename
            num = int(filename.replace('box_', '').replace('.png', ''))

            # Read the label from the CSV
            label = labels.loc[num, 'label']

            # Map the label to a class label
            if label =='RBC':
                class_label = 'rbc'
            elif label =='WBC':
                class_label = 'wbc'
            elif label =='PLT':
                class_label = 'plt' 
            elif label =='AGG':
                class_label = 'agg'
            elif label =='OOF':
                class_label = 'oof'
            else:
                continue
        else:
            continue

        # Create the full path to the image file
        file_path = os.path.join(directory, filename)

        # Calculate features for this image
        features = calculate_contour_properties(file_path, return_list=True)

        # Append features to the respective class list
        class_features[class_label].append(features)

    # Calculate average features for each class
    average_features = {}
    for class_label, features in class_features.items():
        # Check if the list is not empty
        if features:
            average_features[class_label] = np.mean(features, axis=0)

    # Convert the average features to a DataFrame
    df = pd.DataFrame.from_dict(average_features, orient='index', 
                                columns=['cx', 'cy', 'area', 'perimeter', 'circularity', 'average_pixel_value', 'uniformity'])

    # Return the DataFrame
    return df



# if __name__ == "__main__":
    
#     USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
#     image_name = '0080'
#     average_features = calculate_average_features(USER_FOLDER_PATH, image_name)
#     # average_features.to_c
#     print(average_features)
