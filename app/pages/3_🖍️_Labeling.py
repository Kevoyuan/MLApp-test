import pickle
import numpy as np
from streamlit_toggle import st_toggle_switch
import shutil
import sys

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox
from css_style import generate_progress_bar, generate_count_bar

from streamlit_image_coordinates import streamlit_image_coordinates
from username_util import handle_username, create_directory_if_not_exists

from PIL import Image, ImageDraw
import os
import pandas as pd

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


def delete_file(file_path):
    """Delete a file at the given path."""
    if os.path.isfile(file_path):
        os.remove(file_path)
        st.success("File removed successfully.")
    else:
        st.error("File not found.")


def delete_folder(folder_path):
    """Delete a folder at a specified path."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def delete_image(image_folder):
    """Allow users to delete an image from the image folder."""

    # List all image files in the directory
    image_files = [f for f in os.listdir(
        image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

    # Let the user select an image to delete
    selected_image = selectbox("Select an image to delete", image_files)

    if selected_image is not None:
        # Delete the selected image
        image_path = os.path.join(image_folder, selected_image)
        if st.sidebar.button('Delete Image'):
            delete_file(image_path)
            delete_file(f'labeled_mask/{image_name}.csv')
            delete_folder(f'data/{image_name}')


def is_point_in_mask(point, mask):
    """Check if a point is in a mask."""
    x, y = point
    height = len(mask)
    width = len(mask[0]) if height > 0 else 0
    if 0 <= x < width and 0 <= y < height:
        return mask[y][x] == 1
    else:
        return False


def change_label(mask, new_label, df, csv_path, mask_name):
    """Change the label of the given mask to the new label and save the changes to a CSV file."""
    # Find the index of the row that corresponds to the given mask
    mask_index = None
    for i, row_mask in enumerate(df['masks']):
        if np.array_equal(row_mask, mask):
            mask_index = i
            break

    # Update the 'label' column for the corresponding row
    df.at[mask_index, 'label'] = new_label

    # Save the DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)
    # labeled_mask(mask_name, csv_path)


@st.cache_data
def labeled_mask(mask_name, csv_name):
    # Read the pickle file
    with open(mask_name, 'rb') as f:
        masks = pickle.load(f)

    # Load the CSV data
    df_csv = pd.read_csv(csv_name, header=0)
    df_csv['label'] = df_csv['label'].fillna('unlabeled')
    print(df_csv)

    # Initialize an empty list to store the combined data
    combined_data = []

    # Iterate over the masks and labels
    for i in range(len(masks)):
        # Check if the index exists in the DataFrame
        if i in df_csv.index:
            label = df_csv['label'].iloc[i]
        else:
            label = 'default_value'  # Or some other default value

        # Create a dictionary with the mask and label
        data_dict = {'mask': masks[i], 'label': label}
        combined_data.append(data_dict)

    return combined_data

def styled_button(col, label, df, csv_path, mask_name):
    """Create a styled button. When the button is clicked, change the label of the highlighted cell to the button's label."""
    # If the current button label matches the last clicked button label, turn it green
    if st.session_state.get("last_clicked_button") == label:
        button_color = '🐤'  # green
    # If the current button's label matches the label of the point in the image that was last clicked, turn it green
    elif label is not None and st.session_state["points"]:
        button_color = '🤍'  # default to white
        for mask_label, masks in label_lists.items():
            for mask in masks:
                if is_point_in_mask(st.session_state["points"][-1], mask) and mask_label == f'list_masks_{label}':
                    button_color = '🐤'  # green
                    break
    else:
        button_color = '🤍'  # white

    # Create the button
    button_clicked = col.button(f'{button_color} {label}')

    # If the button is clicked, change the label of the selected mask to this button's label and update the last clicked button
    if button_clicked and st.session_state["selected_mask"] is not None:
        st.session_state['last_clicked_button'] = label
        # Pass the label as a string
        change_label(st.session_state["selected_mask"], str(
            label), df, csv_path, mask_name)

        # Reset the selected mask
        st.session_state["selected_mask"] = None
        st.experimental_rerun()

    # If the button is not clicked but a mask in the image is selected, reset the last clicked button
    elif not button_clicked and st.session_state["selected_mask"] is not None:
        st.session_state['last_clicked_button'] = None


@st.cache_data
def merge_csv(directory):
    # Use os.listdir to get all files in the directory
    all_files = os.listdir(directory)

    # Filter the list down to only csv files
    csv_files = [file for file in all_files if file.endswith('.csv')]

    # Initialize an empty list to hold dataframes
    df_list = []

    # Loop over all csv files and read each one into a pandas DataFrame
    for file in csv_files:
        path = os.path.join(directory, file)  # Get full path to the CSV file
        df = pd.read_csv(path)
        df_list.append(df)

    # Concatenate all the dataframes together
    merged_df = pd.concat(df_list)

    # Save the merged dataframe to a new csv file
    merged_df.to_csv('labeled_mask/merged.csv', index=False)


@st.cache_data
def clear_column(csv_path, column_name, mask_name):

    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Clear the values in the specified column
    df[column_name] = None

    # Save the modified DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)

    # df = pd.DataFrame(columns=['masks', 'label'])

    st.session_state['df'] = load_data(mask_name)


def apply_colored_masks(image_path, masks, labels, color_dict):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        for idx, mask in enumerate(masks):
            # Check if the label for the current mask is not NaN
            if len(labels) > idx and not pd.isna(labels[idx]):
                # Convert mask to PIL Image and ensure it's 'L' mode
                mask_img = Image.fromarray(
                    np.array(mask).astype('uint8') * 255).convert('L')

                # Get the label for the current mask
                label = labels[idx]

                # Get the color for the current label
                # Default to white if label not in color_dict
                color = color_dict.get(label, (255, 255, 255))

                # Create overlay image
                overlay = Image.new('RGB', img.size, color)

                # Paste the overlay onto the image using the mask
                img.paste(overlay, mask=mask_img)

        return img
###############################################################


@st.cache_data
def load_data(mask_name):
    # Read the pickle file and create the DataFrame
    with open(mask_name, 'rb') as f:
        masks = pickle.load(f)

    mask_list = []
    for i in range(len(masks)):
        mask_list.append({'masks': masks[i]})
        # process mask and coordinate

    df = pd.DataFrame(mask_list)

    # df.to_csv(csv_path, index=False)
    # st.write(df.columns.values)
    # print(mask_list)

    return df

@st.cache_data
def process_and_get_image_coordinates(box_img, masks_to_color, mask_labels, color_dict):
    img = apply_colored_masks(box_img, masks_to_color, mask_labels, color_dict)
    value = streamlit_image_coordinates(img)
    return value

########################################################
username = handle_username()

# print(username)
user_folder = f'./user_folders/{username}'
create_directory_if_not_exists(user_folder)

# SAVE_ROOT_PATH = '/Volumes/group05/APP_test/dataset'

SAVE_ROOT_PATH = f'{user_folder}/dataset/sam'

# SAVE_ROOT_PATH = '/Volumes/group05/APP_test/dataset'


with st.sidebar:
    image_folder = f'{user_folder}/dataset/original_image/'
    # uploaded_image = upload_image(image_folder)

    image_files = [f for f in os.listdir(
        image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
    # Create a select list with image file options
    selected_image = st.selectbox("Select an image", image_files)
    image_name = os.path.splitext(selected_image)[0]
    csv_path = f'{SAVE_ROOT_PATH}/{image_name}/{image_name}.csv'
    mask_name = f'{SAVE_ROOT_PATH}/{image_name}/segmentation.pkl'
    box_img = f'{SAVE_ROOT_PATH}/{image_name}/bbox.png'
    if not os.path.isfile(mask_name) or not os.path.isfile(box_img):
        st.warning(
            f'The image {image_name} is not segmented, please run the segmentation.')
        sys.exit('Program terminated.')
    # If this is the first time running, or if the selected image has changed, update the session state
    if 'selected_image' not in st.session_state or st.session_state['selected_image'] != selected_image:
        # update the selected image in the session state
        st.session_state['selected_image'] = selected_image

        st.session_state['df'] = load_data(
            mask_name)

    on = st_toggle_switch(
        label="Advance Setting",
        key="switch_1",
        default_value=False,
        label_after=True,
        inactive_color="#D3D3D3",
        active_color="#11567f",
        track_color="#29B5E8",

    )
    if on:
        # Create a Streamlit button
        toggle_button = st.checkbox("Clear labels")

        if toggle_button:
            st.warning(
                "Warning: This process will clear the labels you have marked!")
            button_clicked = st.button("Confirm")

            # Check if the button is clicked
            if button_clicked:
                if not os.path.isfile(csv_path):
                    pass
                else:
                    clear_column(csv_path, 'label', mask_name)

                # st.session_state['df']=pd.DataFrame(columns=['masks', 'label'])

                st.success("The cell label has been cleared.")
        delete_image(image_folder)

# Construct the full path to the selected image
image_path = os.path.join(image_folder, selected_image)

if "points" not in st.session_state:
    st.session_state["points"] = []
if "selected_mask" not in st.session_state:
    st.session_state["selected_mask"] = None


df = st.session_state['df']

if 'label' not in df.columns and not df.empty:
    df['label'] = ''


labels = {'WBC', 'RBC', 'AGG', 'PLT', 'OOF'}
label_lists = {}

# Check if the 'label' column exists in the DataFrame
if 'label' not in df.columns:
    st.image(box_img)
    st.success("No cells detected.")
    # You can choose to exit the script or perform any other desired action
else:
    unique_labels = df['label'].unique()

    if len(unique_labels) > 0:
        for label in unique_labels:
            masks = df.loc[df['label'] == label, 'masks'].tolist()
            list_name = f"list_masks_{label}"
            label_lists[list_name] = masks

    # Check if the csv file exists
    if not os.path.isfile(csv_path):
        # If the file doesn't exist, create a new DataFrame with the appropriate columns
        mask_label_frame = pd.DataFrame(columns=['masks', 'label'])
        # Save the DataFrame as a CSV file
        mask_label_frame.to_csv(csv_path, index=False)
    else:
        # If the file exists, read it
        mask_label_frame = pd.read_csv(csv_path)

    # Define a dictionary that maps labels to colors
    color_dict = {
        'WBC': (89, 75, 110),  # White
        'RBC': (255, 0, 0),      # Red
        'AGG': (0, 0, 255),      # Blue
        'PLT': (255, 140, 0),    # Orange
        'OOF': (128, 0, 128)     # Purple
    }

    mask_labels = mask_label_frame['label']
    masks_to_color = df['masks']

    col1, col2 = st.columns([5, 1])

    with col1:
        img = apply_colored_masks(
            box_img, masks_to_color, mask_labels, color_dict)
        value = streamlit_image_coordinates(img)
############################################################################

    # initial point coordinate
    point = 0, 0
    if value is not None:

        point = value["x"], value["y"]

        st.session_state["points"].append(point)

        for label, masks in label_lists.items():
            for mask in masks:

                if is_point_in_mask(point, mask):
                    st.session_state["selected_mask"] = mask
                    st.session_state['df'] = df
                    break

    with st.sidebar:
        if on:

            st.write("cursor: ", point)

    ############################################################################

    # count of unlabeled masks
    unlabeled_count = mask_label_frame['label'].isna().sum()
    labeled_count = mask_label_frame['label'].value_counts().sum()

    # st.write(labeled_count)
    total_count = len(mask_label_frame)  # total count of masks
    # example gradient
    with col2:
        # add_vertical_space(5)
        gradient = "linear-gradient(#94FAF0 , #31D1D0)"

        generate_count_bar("labeled", labeled_count, total_count, gradient)

    # # Create buttons for each cell
    cell_types = ["WBC", "RBC", "PLT", "AGG", "OOF"]
    columns = st.columns(7)
    for cell_type, col in zip(cell_types, columns):
        styled_button(col, cell_type, df, csv_path, mask_name)

    mask_label = pd.read_csv(csv_path)

    label_counts = mask_label['label'].value_counts()

    complete_label_counts = pd.Series(0, index=labels)
    complete_label_counts = complete_label_counts.add(
        label_counts, fill_value=0)

    # save the mask + Label data to npy file
    combined_data = labeled_mask(mask_name, csv_path)
    npy_file = f'{SAVE_ROOT_PATH}/{image_name}/{image_name}.npy'
    np.save(npy_file, combined_data)

    from detection import save_boxes_from_npy

    # save_boxes_from_npy(user_folder, npy_file)

    add_vertical_space(2)
    ###########################################################################

    # Bar chart

    # initial data
    data = complete_label_counts.reset_index()
    data.columns = ['Label', 'Count']
    df_count = data['Count']
    wbc_count = data.loc[data['Label'] == 'WBC', 'Count'].iloc[0]
    rbc_count = data.loc[data['Label'] == 'RBC', 'Count'].iloc[0]
    plt_count = data.loc[data['Label'] == 'PLT', 'Count'].iloc[0]
    agg_count = data.loc[data['Label'] == 'AGG', 'Count'].iloc[0]
    oof_count = data.loc[data['Label'] == 'OOF', 'Count'].iloc[0]

    generate_progress_bar(
        "WBC", df_count, wbc_count, "linear-gradient(to right, #a0a5b9 0%, #cfd9df 100%)")
    generate_progress_bar(
        'RBC', df_count, rbc_count, "linear-gradient(to right, #e3e7eb, #cfd9df)")
    generate_progress_bar(
        'PLT', df_count, plt_count, "linear-gradient(to right, #cfd9df, #a0a5b9)")
    generate_progress_bar(
        'AGG', df_count, agg_count, "linear-gradient(to right, #a0a5b9, #e3e7eb)")
    generate_progress_bar(
        'OOF', df_count, oof_count, "linear-gradient(to right, #cfd9df, #a0a5b9)")


###############################################################


# Create three columns
# col1, col2, col3 = st.columns([2, 2, 1])
if st.sidebar.button("🤙🏻 Summit"):
    merge_csv('labeled_mask')

    switch_page("Classification")
