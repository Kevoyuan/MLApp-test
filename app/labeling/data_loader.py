import streamlit as st
import os
import pandas as pd
import pickle
from setup.config import login_statement, user_folder_config
import sys
import glob
USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()

save_root = SAVE_ROOT_PATH
# print(f"data_loader: user_folder: {USER_FOLDER_PATH}\n")


@st.cache_data
def load_mask(mask_path):
    """
    Loads a mask from a pickle file.

    This function opens a pickle file at the specified path and loads the mask contained within it.

    Args:
        mask_path (str): The path to the pickle file containing the mask.

    Returns:
        masks: The mask loaded from the pickle file.
    """
    with open(mask_path, "rb") as f:
        masks = pickle.load(f)
    return masks

@st.cache_data
def load_data(mask_from_pkl):
    """
    Loads mask data from a pickle file and creates a DataFrame.

    This function opens a pickle file at the specified path, loads the masks contained within it,
    and creates a DataFrame where each row corresponds to a mask.

    Args:
        mask_path (str): The path to the pickle file containing the masks.

    Returns:
        df (pandas.DataFrame): A DataFrame where each row contains a mask from the pickle file.
    """
    masks = mask_from_pkl

    mask_list = []
    for i in range(len(masks)):
        mask_list.append({"masks": masks[i]})
        # process mask and coordinate

    df = pd.DataFrame(mask_list)

    return df

def get_image_files(user_folder):
    """
    Get all image files in a user's folder.

    This function looks for all files in the 'dataset/original/' subdirectory
    of the specified user folder that have a .png, .jpg, or .jpeg extension.
    If the image folder does not exist, an error message is displayed.
    If there are no image files in the folder, a warning message is displayed.

    Parameters:
    user_folder (str): The path to the user's folder.

    Returns:
    tuple: A tuple containing the path to the image folder and a list of image files.
           If the image folder does not exist or there are no image files, returns (None, None).
    """
    image_folder = f"{user_folder}/dataset/original/"
    if not os.path.exists(image_folder):
        st.error(f"The folder {image_folder} does not exist.")
        return None, None
    image_files = [
        f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    
    # def sorting_key(filename):
    #     base_name = filename.split(".")[0]
    #     return int(base_name) if base_name.isdigit() else base_name

    image_files = sorted(image_files)

    if not image_files:
        st.warning(f"There is no image file in the user folder.")
        st.warning("Please back to the Segmentation page.")
        sys.exit("Program terminated.")

    return image_folder, image_files


def get_or_create_mask_label_df(csv_path):
    """
    Get or create a DataFrame for mask labels.

    This function checks if a CSV file exists at the given path. If the file exists,
    it is read into a DataFrame. If the file does not exist, a new DataFrame is created
    with columns for 'masks' and 'label', and this DataFrame is saved as a CSV file at
    the given path.

    Parameters:
    csv_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: A DataFrame containing mask labels.
    """
    # Check if the csv file exists
    if not os.path.isfile(csv_path):
        # If the file doesn't exist, create a new DataFrame with the appropriate columns
        mask_label_df = pd.DataFrame(columns=["masks", "label"])
        try:
            # Save the DataFrame as a CSV file
            mask_label_df.to_csv(csv_path, index=False)
        except Exception as e:
            print("An error occurred:", e)

    else:
        # If the file exists, read it
        mask_label_df = pd.read_csv(csv_path)

    return mask_label_df

@st.cache_data
def construct_file_paths(user_folder, image_name,save_root):
    """
    Constructs the file paths for the CSV file, mask pickle file, and bounding box image associated with a given image.

    This function constructs the paths to the CSV file, mask pickle file, and bounding box image that are associated
    with a specific image in a user's folder.

    Args:
        user_folder (str): The name of the user's folder.
        image_name (str): The name of the image.

    Returns:
        csv_path (str): The path to the CSV file associated with the image.
        mask_path (str): The path to the mask pickle file associated with the image.
        box_img (str): The path to the bounding box image associated with the image.
    """
    csv_path = f"{save_root}/{image_name}/{image_name}.csv"
    mask_path = f"{save_root}/{image_name}/segmentation.pkl"
    box_img = f"{save_root}/{image_name}/bbox.png"
    return csv_path, mask_path, box_img

def select_image(user_folder):
    """
    Allow users to select an image from the image folder.

    Parameters:
    image_folder (str): The path to the image folder.

    Returns:
    str: The path to the selected image.
    """

    # try:
    #     selected_image = st.selectbox("Select an image", image_files)
    # except Exception as e:
    #     # st.error(f"An error occurred: {e}")
    #     # selected_image = None
    #     st.experimental_rerun
    path_pattern = f"{user_folder}/dataset/sam/**/bbox.png"
    bbox_files = glob.glob(path_pattern, recursive=True)
    # Check if the pattern matches any files
    if len(bbox_files) == 0:
        st.info("Please use SAM to segment blood cell images.")
        return None, None

    image_names = ([os.path.basename(os.path.dirname(f)) for f in bbox_files])

    # Create a dictionary to map image names to their corresponding file paths
    image_paths_dict = dict(zip(image_names, bbox_files))

    with st.sidebar:
        image_names = sorted(image_paths_dict.keys())

        selected_image_name = st.selectbox("Background image:", image_names)

    return selected_image_name