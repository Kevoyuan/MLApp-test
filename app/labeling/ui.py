import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from css_style import generate_progress_bar, generate_count_bar
from labeling.mask_operations import is_point_in_mask



# def styled_button(_col, label, df, csv_path,label_lists):
#     """
#     Create a styled button. When the button is clicked, changes the label of the highlighted cell to the button's label.

#     Parameters:
#     label (str): The label for the button.
#     active_color (str): The color of the button when active.
#     inactive_color (str): The color of the button when inactive.
#     button_clicked (bool): Whether the button has been clicked.

#     Returns:
#     bool: Whether the button has been clicked.
#     """  # If the current button label matches the last clicked button label, turn it green
#     # print("\style button 1: \n", df)
#     # print("\n")
#     if st.session_state.get("last_clicked_button") == label:
#         button_color = "🐤"  # green
#     # If the current button's label matches the label of the point in the image that was last clicked, turn it green
#     elif label is not None and st.session_state["points"]:
#         button_color = "🤍"  # default to white
#         for mask_label, masks in label_lists.items():
#             for mask in masks:
#                 if (
#                     is_point_in_mask(st.session_state["points"][-1], mask)
#                     and mask_label == f"list_masks_{label}"
#                 ):
#                     button_color = "🐤"  # green
#                     break
#     else:
#         button_color = "🤍"  # white

#     label_colors = {
#         "WBC": "White blood cells",
#         "RBC": "Red blood cells",
#         "AGG": "Aggregation of several cells of the previous groups, i.e. they are sticking together",
#         "PLT": "Platelets (also known as thrombocytes)",
#         "OOF": "Out of focus, blurry cells, happens when a cell floats by above or below the focal plane of the microscope",
#     }
#     # Create the button
#     with col:
#         button_clicked = st.button(
#             f"{button_color} {label}", help=label_colors.get(label, "")
#         )

#     try:
#         # If the button is clicked, change the label of the selected mask to this button's label and update the last clicked button
#         if button_clicked and st.session_state["selected_mask"] is not None:
#             st.session_state["last_clicked_button"] = label
#             # Pass the label as a string
#             df = change_label(
#                 st.session_state["selected_mask"],
#                 st.session_state["last_clicked_button"],
#                 df,
#                 csv_path,
#             )
#             st.session_state["df"] = df
#             # Reset the selected mask
#             st.session_state["selected_mask"] = None
#             st.experimental_rerun()

#         # If the button is not clicked but a mask in the image is selected, reset the last clicked button
#         elif not button_clicked and st.session_state["selected_mask"] is not None:
#             st.session_state["last_clicked_button"] = None
#     except Exception as e:
#         print("An error occurred:", e)
        
def generate_labeled_count_bar(mask_label_df):
    """
    Generates a count bar for labeled and unlabeled masks in a DataFrame.

    This function calculates the number of labeled and unlabeled masks in a DataFrame, and then generates a count bar
    using a custom gradient. The count bar represents the proportion of labeled masks out of the total number of masks.

    Args:
        mask_label_df (pandas.DataFrame): A DataFrame containing the labels for each mask.

    """
    # count of unlabeled masks
    unlabeled_count = mask_label_df["label"].isna().sum()
    labeled_count = mask_label_df["label"].value_counts().sum()

    # st.write(labeled_count)
    total_count = len(mask_label_df)  # total count of masks

    # example gradient
    gradient = "linear-gradient(#94FAF0 , #31D1D0)"

    generate_count_bar("labeled", labeled_count, total_count, gradient)
    
    
@st.cache_resource(show_spinner=False)
def process_labels_and_generate_bars(mask_label_df, labels):
    """
    Processes the labels in a DataFrame and generates progress bars for each label.

    This function calculates the count of each label in a DataFrame, completes the label counts with zero for missing labels,
    and then generates a progress bar for each label using a custom gradient.

    Args:
        mask_label_df (pandas.DataFrame): A DataFrame containing the labels for each mask.
        labels (list): A list of all possible labels.

    """
    label_counts = mask_label_df["label"].value_counts()

    complete_label_counts = pd.Series(0, index=labels)
    complete_label_counts = complete_label_counts.add(label_counts, fill_value=0)

    ###########################################################################

    # Bar chart

    # initial data
    data = complete_label_counts.reset_index()
    data.columns = ["Label", "Count"]
    df_count = data["Count"]

    # Define a dictionary that maps labels to gradients
    gradient_dict = {
        "WBC": "linear-gradient(to right, #a0a5b9 0%, #cfd9df 100%)",
        "RBC": "linear-gradient(to right, #e3e7eb, #cfd9df)",
        "PLT": "linear-gradient(to right, #cfd9df, #a0a5b9)",
        "AGG": "linear-gradient(to right, #a0a5b9, #e3e7eb)",
        "OOF": "linear-gradient(to right, #cfd9df, #a0a5b9)",
    }

    # Generate progress bars for each label
    for label, gradient in gradient_dict.items():
        label_count = data.loc[data["Label"] == label, "Count"].iloc[0]
        generate_progress_bar(label, df_count, label_count, gradient)


def apply_colored_masks(image_path, masks, labels, color_dict):
    """
    Apply colored masks to an image.

    Parameters:
    image_path (str): The path to the image file.
    masks (list): A list of masks to apply.
    labels (list): A list of labels corresponding to the masks.
    color_dict (dict): A dictionary mapping labels to colors.

    Returns:
    PIL.Image: The image with the masks applied.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        for idx, mask in enumerate(masks):
            # Check if the label for the current mask is not NaN
            if len(labels) > idx and not pd.isna(labels[idx]):
                # Convert mask to PIL Image and ensure it's 'L' mode
                mask_img = Image.fromarray(
                    np.array(mask).astype("uint8") * 255
                ).convert("L")

                # Get the label for the current mask
                label = labels[idx]

                # Get the color for the current label
                # Default to white if label not in color_dict
                color = color_dict.get(label, (255, 255, 255))

                # Create overlay image
                overlay = Image.new("RGB", img.size, color)

                # Paste the overlay onto the image using the mask
                img.paste(overlay, mask=mask_img)

        return img

@st.cache_data
def show_image_in_expander():
    with st.expander("Example of different cell types",expanded=True):
        st.image("example_image/example.png")
        
