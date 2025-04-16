import streamlit as st
from labeling.mask_operations import is_point_in_mask
from setup.config import login_statement, user_folder_config
from setup.user_util import get_directory_size,create_directory_if_not_exists
import numpy as np
import os
import pandas as pd
import shutil

st.cache_resource.clear()

USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
save_root = SAVE_ROOT_PATH
# print(f"data processor: user_folder: {USER_FOLDER_PATH}\n")

def process_point(coordinates, df):
    """
    Process a point and update the session state.

    This function checks if a point is within any of the masks in label_lists.
    If the point is within a mask, the mask and DataFrame are added to the session state.

    Parameters:
    coordinates (dict): A dictionary containing the x and y coordinates of the point.
    df (pandas.DataFrame): The DataFrame to be added to the session state if the point is in a mask.

    Returns:
    None
    """
    # initial point coordinate
    point = 0, 0
    if coordinates is not None:
        # print("coordinates: ", coordinates)
        point = coordinates["x"], coordinates["y"]
        st.session_state["points"].append(point)

        for i, mask in enumerate(df["masks"]):
            if is_point_in_mask(point, mask):
                st.session_state["selected_mask"] = mask
                st.session_state["df"] = df
                # Save the label of the selected mask
                st.session_state["selected_label"] = df.at[i, "label"]
                break
            
def process_and_save_mask_data(mask_from_pkl, mask_label_df, image_name,save_root):
    """
    Function to load mask and label data, combine it, and save it into a .npy file.

    Parameters:
    mask_path (str): The path to the mask file.
    csv_path (str): The path to the CSV file.
    SAVE_ROOT_PATH (str): The root path where the .npy file will be saved.
    image_name (str): The name of the image file.

    Returns:
    None
    """

    # Part 1: Load and combine the mask and label data

    # Read the pickle file
    masks = mask_from_pkl

    # Load the CSV data
    df_csv = mask_label_df
    df_csv["label"] = df_csv["label"].fillna("unlabeled")
    # print(df_csv)

    # Initialize an empty list to store the combined data
    combined_data = []

    # Iterate over the masks and labels
    for i in range(len(masks)):
        # Check if the index exists in the DataFrame
        if i in df_csv.index:
            label = df_csv["label"].iloc[i]
        else:
            label = "unlabeled"  # Or some other default value

        # Create a dictionary with the mask and label
        data_dict = {"mask": masks[i], "label": label}
        combined_data.append(data_dict)
    # print("combined_data: \n", combined_data)

    # Part 2: Save the combined data to a .npy file

    # npy_folder_path = f"{SAVE_ROOT_PATH}/npy_file"
    npy_folder_path = f"{save_root}/npy_file"
    

    # Create the directory if it doesn't exist
    create_directory_if_not_exists(npy_folder_path)

    npy_file = f"{npy_folder_path}/{image_name}.npy"
    np.save(npy_file, combined_data)
    

    
def delete_image(image_folder,save_root):
    """
    Allow users to delete an image from the image folder.

    Parameters:
    image_folder (str): The path to the image folder.

    Returns:
    None
    """

    # List all image files in the directory
    image_files = [
        f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files = sorted(image_files)
    # Let the user select an image to delete
    selected_image = st.selectbox("Select an image to delete", image_files)
    
    if selected_image is not None:
        selected_image_name = selected_image.split(".")[0]
        print(f"image_name: {selected_image_name}")

    if selected_image is not None:
        # folder_size = get_directory_size(f"{SAVE_ROOT_PATH}/{selected_image_name}")
        folder_size = get_directory_size(f"{save_root}/{selected_image_name}")
        
        st.write(
            f'The image folder "{selected_image_name}" is {folder_size / (1024 * 1024):.2f} MB'
        )
        # Delete the selected image
        image_path = os.path.join(image_folder, selected_image)
        if st.sidebar.button("Delete Image"):
            # os.remove(f"{SAVE_ROOT_PATH}/npy_file/{selected_image_name}.npy")
            try:
                os.remove(f"{save_root}/npy_file/{selected_image_name}.npy")
            except Exception as e:
                pass
            

            delete_path(image_path)
            # delete_file(f'labeled_mask/{image_name}.csv')
            # delete_folder(f'{SAVE_ROOT_PATH}/{selected_image_name}')
            # delete_path(f"{SAVE_ROOT_PATH}/{selected_image_name}")
            delete_path(f"{save_root}/{selected_image_name}")
            
            st.experimental_rerun()
def delete_path(path):
    """
    Delete a file or a directory at the given path.

    Parameters:
    path (str): The path to the file or directory to be deleted.

    Returns:
    None
    """
    if os.path.isfile(path):
        os.remove(path)
        st.success("File removed successfully.")
    elif os.path.isdir(path):
        shutil.rmtree(path)
        st.success("Directory removed successfully.")
    else:
        st.error("Path not found.")            

def change_label(mask, new_label, df, csv_path):
    """
    Change the label of a mask in a DataFrame.

    Parameters:
    mask (numpy.array): The mask whose label should be changed.
    new_label (str): The new label for the mask.
    df (pandas.DataFrame): The DataFrame containing the mask labels.
    csv_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The DataFrame with the mask's label changed.
    """
    # Find the index of the row that corresponds to the given mask
    mask_index = None
    for i, row_mask in enumerate(df["masks"]):
        if np.array_equal(row_mask, mask):
            mask_index = i
            break
    # print("\nchange label before: \n", df)
    # print("\n")
    # Update the 'label' column for the corresponding row
    df.at[mask_index, "label"] = new_label
    # print("\nchange label after: \n", df)
    # print("\n")

    # Save the DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)
    # labeled_mask(mask_path, csv_path)
    # st.session_state['df'] = df
    return df