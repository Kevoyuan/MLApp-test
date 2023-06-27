import pickle
import numpy as np
from streamlit_toggle import st_toggle_switch

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox

from streamlit_image_coordinates import streamlit_image_coordinates

from PIL import Image, ImageDraw
import os
import pandas as pd

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)


def delete_file(file_path):
    """Delete a file at the given path."""
    if os.path.isfile(file_path):
        os.remove(file_path)
        st.success("File removed successfully.")
    else:
        st.error("File not found.")


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


def is_point_in_mask(point, mask):
    """Check if a point is in a mask."""
    x, y = point
    height = len(mask)
    width = len(mask[0]) if height > 0 else 0
    if 0 <= x < width and 0 <= y < height:
        return mask[y][x] == 1
    else:
        return False


def draw_mask_if_point_in_mask(img, point, mask):
    if is_point_in_mask(point, mask):
        mask_indices = np.where(mask == 1)
        img[mask_indices] = [255, 0, 0]  # change masked pixels to red

    return img


def change_label(mask, new_label, df, csv_path):
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


def styled_button(col, text, label, df, csv_path):
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
    button_clicked = col.button(f'{button_color} {text}')

    # If the button is clicked, change the label of the selected mask to this button's label and update the last clicked button
    if button_clicked and st.session_state["selected_mask"] is not None:
        st.session_state['last_clicked_button'] = label
        # Pass the label as a string
        change_label(st.session_state["selected_mask"], str(
            label), df, csv_path)
        # Reset the selected mask
        st.session_state["selected_mask"] = None
        st.experimental_rerun()

    # If the button is not clicked but a mask in the image is selected, reset the last clicked button
    elif not button_clicked and st.session_state["selected_mask"] is not None:
        st.session_state['last_clicked_button'] = None


def generate_progress_bar(label, count, gradient):
    display_count = count/(data['Count'].max()+1)*100+10
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    .flex-container {{
        display: flex;
        align-items: center;
        margin-bottom: -5px;
        justify-content: flex-start;
        margin-top: 0px
    }}

    .title {{
        width: 40px; 
        margin-right: 10px;
        margin: -5px;
        # display: flex;       
        align-items: center;
        font-size: 16px;
        # font-weight: bold;
        text-align: left; 
        color: #333; 
    }}

    .container {{
        background-color: #e3e7eb;
        width: 80%;  
        # flex-grow: 1;
        height: 18px; 
        border-radius: 20px;
        margin-bottom: 0px
    }}

    .text-{label} {{
        background-image: {gradient};
        color: white;
        padding: 0.1%;
        text-align: right;
        font-size: 22px;
        font-weight: bold; 
        text-shadow: 2px 2px 8px #000000; 
        border-radius: 22px;
        height: 100%;
        line-height: 15px;
        width: {display_count}%;  
        animation: progress-bar-width-{label} 1.5s ease-out 1;
        transition: width 1.2s ease-out;
        margin-top: 0px
        margin-bottom: 0px
    }}

    .percent-{label} {{
        width: {display_count}%;
    }}

    @keyframes progress-bar-width-{label} {{
        0% {{ width: 0; }}
        100% {{ width: {display_count}%; }}  
    }}
    </style>
    """

    # Inject the CSS into the Streamlit app
    st.markdown(css, unsafe_allow_html=True)

    # Create progress bar with custom CSS
    st.markdown(f"""
    <div class="flex-container">
        <div class="title">{label}</div>
        <div class="container">
            <div class="text-{label} percent-{label}">{int(count)}</div> 
        </div>
    </div>
    """, unsafe_allow_html=True)


def generate_count_bar(label, count, total, gradient):
    if total != 0:
        display_count = count / total * 100
    else:
        display_count = 0  # or whatever value you want to assign when total is zero

    css = f"""
    <style>
    @font-face {{
        font-family: "San Francisco";
        font-weight: 400;
        src: url("https://applesocial.s3.amazonaws.com/assets/styles/fonts/sanfrancisco/sanfranciscodisplay-regular-webfont.woff");
    }}

    .vertical-bar-container {{
        height: 300px;
        width: 30px;
        display: flex;
        flex-direction: column-reverse;
        align-items: center;
        background-color: #f2f2f2;
        border-radius: 10px;
        padding: 5px;
        margin-right: 10px;
    }}

    .vertical-bar {{
        background-image: {gradient};
        width: 100%;
        border-radius: 10px;
        transition: height 1s ease-in-out;
    }}

    .percentage {{
        margin-top: 5px;
        font-size: 14px;
        font-weight: bold;
        font-family: "San Francisco", -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .label {{
        font-size: 12px;
        color: #555;
        font-family: "San Francisco", -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="vertical-bar-container">
        <div class="vertical-bar" style="height: {display_count}%;"></div>
    </div>
    <div class="percentage">{display_count:.1f}%</div>
    <div class="label">{label}</div>
    """, unsafe_allow_html=True)


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


def clear_column(csv_path, column_name, mask_name):
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Clear the values in the specified column
    df[column_name] = None

    # Save the modified DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)

    # df = pd.DataFrame(columns=['masks', 'label'])

    st.session_state['df'], data = load_data(mask_name)


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


def load_data(mask_name):
    # Read the pickle file and create the DataFrame
    with open(mask_name, 'rb') as f:
        masks = pickle.load(f)

    data = []
    for i in range(len(masks)):
        data.append({'masks': masks[i]})
        # process mask and coordinate

    df = pd.DataFrame(data)
    return df, data


########################################################
image_folder = "image/"
# uploaded_image = upload_image(image_folder)

image_files = [f for f in os.listdir(
    image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]


with st.sidebar:
    # Create a select list with image file options
    selected_image = st.selectbox("Select an image", image_files)
    image_name = os.path.splitext(selected_image)[0]

    csv_path = f'labeled_mask/{image_name}.csv'
    mask_name = f'image/{image_name}.pkl'
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


if 'df' not in st.session_state:
    st.session_state['df'], data = load_data(mask_name)

df = st.session_state['df']

if 'label' not in df.columns and not df.empty:
    df['label'] = ''


labels = {'WBC', 'RBC', 'AGG', 'PLT', 'OOF'}
label_lists = {}

unique_labels = df['label'].unique()  # This will ignore NaN values
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
        image_path, masks_to_color, mask_labels, color_dict)
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


# st.write(is_point_in_mask(point, masks[0]))


# # Create buttons for each cell
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)


styled_button(col1, "WBC", "WBC", df, csv_path)
styled_button(col2, "RBC", "RBC", df, csv_path)
styled_button(col3, 'PLT', 'PLT', df, csv_path)
styled_button(col4, "AGG", "AGG", df, csv_path)
styled_button(col5, "OOF", "OOF", df, csv_path)

mask_label = pd.read_csv(csv_path)

label_counts = mask_label['label'].value_counts()
complete_label_counts = pd.Series(0, index=labels)
complete_label_counts = complete_label_counts.add(label_counts, fill_value=0)

# st.write(complete_label_counts)


###########################################################################

# Bar chart

# initial data
data = complete_label_counts.reset_index()
data.columns = ['Label', 'Count']

wbc_count = data.loc[data['Label'] == 'WBC', 'Count'].iloc[0]
rbc_count = data.loc[data['Label'] == 'RBC', 'Count'].iloc[0]
plt_count = data.loc[data['Label'] == 'PLT', 'Count'].iloc[0]
agg_count = data.loc[data['Label'] == 'AGG', 'Count'].iloc[0]
oof_count = data.loc[data['Label'] == 'OOF', 'Count'].iloc[0]

generate_progress_bar(
    "WBC", wbc_count, "linear-gradient(to right, #a0a5b9 0%, #cfd9df 100%)")
generate_progress_bar(
    'RBC', rbc_count, "linear-gradient(to right, #e3e7eb, #cfd9df)")
generate_progress_bar(
    'PLT', plt_count, "linear-gradient(to right, #cfd9df, #a0a5b9)")
generate_progress_bar(
    'AGG', agg_count, "linear-gradient(to right, #a0a5b9, #e3e7eb)")
generate_progress_bar(
    'OOF', oof_count, "linear-gradient(to right, #cfd9df, #a0a5b9)")


###############################################################

add_vertical_space(2)

# Create three columns
# col1, col2, col3 = st.columns([2, 2, 1])
if st.sidebar.button("🤙🏻 Summit"):
    merge_csv('labeled_mask')
    switch_page("Classification")
