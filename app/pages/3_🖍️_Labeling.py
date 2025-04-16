from streamlit_toggle import st_toggle_switch
import sys
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from css_style import color_dict
from streamlit_image_coordinates import streamlit_image_coordinates
import os
import pandas as pd
from setup.config import login_statement, user_folder_config
from labeling.data_loader import (
    load_mask,
    load_data,
    get_image_files,
    get_or_create_mask_label_df,
    construct_file_paths,
    select_image,
)
from labeling.data_processor import (
    process_point,
    process_and_save_mask_data,
    delete_image,
    change_label,
)
from labeling.ui import (
    generate_labeled_count_bar,
    show_image_in_expander,
    process_labels_and_generate_bars,
    apply_colored_masks,
)
from labeling.mask_operations import is_point_in_mask


def styled_button(_col, label, df, csv_path, label_lists):
    """
    Create a styled button. When the button is clicked, changes the label of the highlighted cell to the button's label.

    Parameters:
    label (str): The label for the button.
    active_color (str): The color of the button when active.
    inactive_color (str): The color of the button when inactive.
    button_clicked (bool): Whether the button has been clicked.

    Returns:
    bool: Whether the button has been clicked.
    """
    # if st.session_state.get("last_clicked_button") == label:
    #     button_color = "?"  # green
    # If the current button's label matches the label of the point in the image that was last clicked, turn it green
    if label is not None and st.session_state["points"]:
        button_color = "🤍"  # default to white
        for mask_label, masks in label_lists.items():
            for mask in masks:
                if (
                    is_point_in_mask(st.session_state["points"][-1], mask)
                    and mask_label == f"list_masks_{label}"
                ):
                    button_color = "🐤"  # green
                    break
    else:
        button_color = "🤍"  # white

    label_colors = {
        "WBC": "White blood cells",
        "RBC": "Red blood cells",
        "AGG": "Aggregation of several cells of the previous groups, i.e. they are sticking together",
        "PLT": "Platelets (also known as thrombocytes)",
        "OOF": "Out of focus, blurry cells, happens when a cell floats by above or below the focal plane of the microscope",
    }
    # Create the button
    with _col:
        button_clicked = st.button(
            f"{button_color} {label}",
            help=label_colors.get(label, ""),
            use_container_width=True,
        )

    try:
        # If the button is clicked, change the label of the selected mask to this button's label and update the last clicked button
        if button_clicked and st.session_state["selected_mask"] is not None:
            st.session_state["last_clicked_button"] = label
            # Pass the label as a string
            df = change_label(
                st.session_state["selected_mask"],
                st.session_state["last_clicked_button"],
                df,
                csv_path,
            )
            st.session_state["df"] = df
            # Reset the selected mask
            st.session_state["selected_mask"] = None
            st.experimental_rerun()

        # If the button is not clicked but a mask in the image is selected, reset the last clicked button
        elif not button_clicked and st.session_state["selected_mask"] is not None:
            st.session_state["last_clicked_button"] = None
    except Exception as e:
        print("An error occurred:", e)


def clear_column(csv_path, column_name, mask_path):
    """
    Clear a specific column in a CSV file and remove corresponding mask files.

    Parameters:
    csv_path (str): The path to the CSV file.
    column_name (str): The name of the column to be cleared.
    mask_path (str): The path to the directory containing the mask files.

    Returns:
    None
    """
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Clear the values in the specified column
    df[column_name] = None

    # Save the modified DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)

    # df = pd.DataFrame(columns=['masks', 'label'])

    st.session_state["df"] = load_data(mask_from_pkl)
    st.experimental_rerun()


###############################################################


@st.cache_data
def check_segmentation(mask_path, box_img, image_name):
    """
    Checks if the segmentation has been performed for a given image.

    This function checks if the mask pickle file and the bounding box image exist for a specific image.
    If either file does not exist, it displays a warning and terminates the program.

    Args:
        mask_path (str): The path to the mask pickle file associated with the image.
        box_img (str): The path to the bounding box image associated with the image.
        image_name (str): The name of the image.

    Raises:
        SystemExit: If either the mask pickle file or the bounding box image does not exist.
    """
    if not os.path.isfile(mask_path) or not os.path.isfile(box_img):
        st.warning(
            f"The image {image_name} is not segmented, please run the segmentation."
        )
        sys.exit("Program terminated.")


########################################################


st.cache_resource.clear()

USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
user_folder = USER_FOLDER_PATH
save_root = SAVE_ROOT_PATH
print(f"Labeling: user_folder: {USER_FOLDER_PATH}\n")

with st.sidebar:
    login_statement()

    # print(user_folder)
    image_folder, image_files = get_image_files(user_folder)
    selected_image = select_image(user_folder)
    image_name = os.path.splitext(selected_image)[0]
    csv_path, mask_path, box_img = construct_file_paths(
        user_folder, image_name, save_root
    )

    mask_label_df = get_or_create_mask_label_df(csv_path)

    check_segmentation(mask_path, box_img, image_name)
    mask_from_pkl = load_mask(mask_path)

    # If this is the first time running, or if the selected image has changed, update the session state
    if (
        "selected_image" not in st.session_state
        or st.session_state["selected_image"] != selected_image
    ):
        # update the selected image in the session state
        st.session_state["selected_image"] = selected_image

        # st.session_state["df"] = load_data(mask_path)
        st.session_state["df"] = load_data(mask_from_pkl)

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
                "Warning: This process will clear the labels you have marked in this image!"
            )
            button_clicked = st.button("Confirm")

            # Check if the button is clicked
            if button_clicked:
                if not os.path.isfile(csv_path):
                    pass
                else:
                    clear_column(csv_path, "label", mask_path)

                # st.session_state['df']=pd.DataFrame(columns=['masks', 'label'])

                st.success("The cell label has been cleared.")

        delete_image(image_folder, save_root)


if "points" not in st.session_state:
    st.session_state["points"] = []
if "selected_mask" not in st.session_state:
    st.session_state["selected_mask"] = []


df = st.session_state["df"]

if "label" not in df.columns and not df.empty:
    df["label"] = ""


labels = {"WBC", "RBC", "AGG", "PLT", "OOF"}
label_lists = {}

# Check if the 'label' column exists in the DataFrame
if "label" not in df.columns:
    st.image(box_img)
    st.success("No cells detected.")
    # You can choose to exit the script or perform any other desired action
else:
    unique_labels = df["label"].unique()

    if len(unique_labels) > 0:
        for label in unique_labels:
            masks = df.loc[df["label"] == label, "masks"].tolist()
            list_name = f"list_masks_{label}"
            label_lists[list_name] = masks

    mask_labels = mask_label_df["label"]
    masks_to_color = df["masks"]
    # Combine the two series into a DataFrame
    styled_button_df = pd.concat([masks_to_color, mask_labels], axis=1)
    # print("styled_button_df:\n", styled_button_df)

    col1, col2 = st.columns([4, 1])

    with col1:
        img = apply_colored_masks(box_img, masks_to_color, mask_labels, color_dict)
        coordinates = streamlit_image_coordinates(img)
        # st.experimental_rerun()
    ############################################################################

    # initial point coordinate get the cursor point and its state
    process_point(coordinates, df)

    with st.sidebar:
        if on:
            try:
                st.write("cursor: ", st.session_state["points"][-1])
            except IndexError:
                x, y = 0, 0
                st.write("cursor: ", (x, y))

            pass

    ############################################################################

    with col2:
        generate_labeled_count_bar(mask_label_df)

    # # Create buttons for each cell
    cell_types = ["RBC", "WBC", "PLT", "AGG", "OOF"]
    columns = st.columns(7)
    # print('mask label df:\n', mask_label_df)
    for cell_type, column in zip(cell_types, columns):
        # print("column: ",column)
        styled_button(column, cell_type, styled_button_df, csv_path, label_lists)

    # generate progressbar and their counter --> bar chart
    process_labels_and_generate_bars(mask_label_df, labels)
    process_and_save_mask_data(mask_from_pkl, mask_label_df, image_name, save_root)


add_vertical_space(2)

show_image_in_expander()

###############################################################

if st.sidebar.button("🤙🏻 Submit"):
    switch_page("Classification")
