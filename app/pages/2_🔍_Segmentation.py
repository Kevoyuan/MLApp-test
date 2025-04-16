import time
import streamlit as st
import randfacts
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import os
from streamlit_extras.switch_page_button import switch_page
from css_style import generate_cork_board, custom_spinner, generate_menu_styles
from setup.user_util import create_directory_if_not_exists
from setup.config import login_statement, user_folder_config
from detection import get_image_masks, predict_from_bbox
from labeling.ui import apply_colored_masks
from streamlit_toggle import st_toggle_switch
from streamlit_option_menu import option_menu
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import glob
import sys
import numpy as np
import cv2

color_dict = {"box": (255, 153, 255)}


def upload_images(image_folder):
    """
    Allows the user to upload multiple image files and saves them to a specified folder.

    This function uses Streamlit's file_uploader to allow the user to upload multiple image files. It then saves these
    images to a specified folder and returns a list of the images and their paths.

    Args:
        image_folder (str): The path to the folder where the images should be saved.

    Returns:
        tuple: A tuple containing a list of the uploaded images and a list of the paths to the saved images.
    """
    uploaded_files = st.sidebar.file_uploader(
        "Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    images = []
    image_paths = []  # This will hold the paths of the saved images

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Convert the file to an image
            image = Image.open(uploaded_file)

            create_directory_if_not_exists(image_folder)

            # Save the image to the image folder
            image_path = os.path.join(image_folder, uploaded_file.name)
            image.save(image_path)
            images.append(image)

            # Append the path of the saved image to the list
            image_paths.append(image_path)

    return images, image_paths


def display_fact_on_board(board_placeholder):
    """
    Displays a random fact on a cork board in a Streamlit app.

    This function uses the randfacts library to get a random fact, then uses a custom markdown generator
    to create a cork board with the fact. The cork board is then displayed in a Streamlit placeholder.

    Args:
        board_placeholder (streamlit.delta_generator.DeltaGenerator): The Streamlit placeholder in which to display the cork board.

    """  # Get a random fact
    fun_info = randfacts.get_fact()

    # Generate the markdown for the cork board with the fact and display it in the placeholder
    board_placeholder.markdown(generate_cork_board(fun_info), unsafe_allow_html=True)


def process_images(save_root, image_paths, board_placeholder):
    """
    Processes a list of images and displays the original and segmented images side by side in a Streamlit app.

    This function iterates over a list of image paths, processes each image, and displays the original and segmented images side by side.
    After processing each image, a random fact is displayed on a cork board if a Streamlit placeholder is provided.

    Args:
        image_paths (list): A list of paths to the images to be processed.
        board_placeholder (streamlit.delta_generator.DeltaGenerator): The Streamlit placeholder in which the cork board should be displayed.
    """
    for i, image_path in enumerate(image_paths):
        # Update the board after each task if the placeholder is not None
        if board_placeholder is not None:
            display_fact_on_board(board_placeholder)

        dir = os.path.splitext(os.path.basename(image_path))[0]
        print("image_path: ", image_path)

        os.makedirs(os.path.join(SAVE_ROOT_PATH, dir), exist_ok=True)
        st.text(os.path.basename(image_path))

        get_image_masks(
            save_root,
            i,
            dir,
            image_path,
            save_as_pkl=True,
            save_annotated=True,
            return_elapsed_time=True,
            return_annotated=True,
        )

        add_vertical_space(1)

        col1, col2 = st.columns(2)
        col1.image(image_path, caption="Original image")
        col2.image(f"{SAVE_ROOT_PATH}/{dir}/segmented.png", caption="Segmented image")
        time.sleep(0.1)
        st.divider()


def segmentation_sam(image_folder, save_root, user_folder):
    create_directory_if_not_exists(image_folder)
    uploaded_image, image_paths = upload_images(image_folder)

    if not uploaded_image or not image_paths:
        st.info("Please upload images")

    if uploaded_image:
        with st.expander("Preview of the images", expanded=True):
            columns = st.columns(4)  # Create 4 columns
            for i, image_path in enumerate(image_paths):
                image = Image.open(image_path)
                with columns[i % 4]:  # Select the column by index
                    st.image(
                        image,
                        caption=os.path.basename(image_path),
                        use_column_width=True,
                    )

        if st.sidebar.button("🧀Segmentaiton"):
            add_vertical_space(1)
            with custom_spinner(
                "Wait for the cell detection...", display_board=True
            ) as board_placeholder:
                process_images(save_root, image_paths, board_placeholder)
                st.success("Done!", icon="✅")

        if st.sidebar.button("🤙🏻 Submit"):
            switch_page("labeling")


def segmentation_hand(
    user_folder,
    drawing_mode="rect",
    stroke_width=1,
    stroke_color="black",
    # realtime_update=True,
):
    path_pattern = f"{user_folder}/dataset/sam/**/bbox.png"
    bbox_files = glob.glob(path_pattern, recursive=True)
    # Check if the pattern matches any files
    if len(bbox_files) == 0:
        st.info("Please use SAM to segment blood cell images.")
        return None, None

    # Extract image names and sort them along with the corresponding bbox_files
    image_names = [os.path.basename(os.path.dirname(f)) for f in bbox_files]

    # Create a dictionary to map image names to their corresponding file paths
    image_paths_dict = dict(zip(image_names, bbox_files))

    with st.sidebar:
        image_names = sorted(image_paths_dict.keys())
        selected_image_name = st.selectbox("Background image:", image_names)
        if st.session_state.get("selected_image_name", "") != selected_image_name:
            # If the selected image has changed, set the new image and rerun the app
            st.session_state["selected_image_name"] = selected_image_name
            # st.experimental_rerun()
        # Get the file path of the selected image
        bg_image = image_paths_dict[selected_image_name]
        print("\nimage_path: ", bg_image)
        background_image = Image.open(bg_image)
        selected_objects_new = []
        selected_objects = []

    col1, col2, col3 = st.columns([1, 5, 1])
    # Create a canvas component

    if background_image is not None:
        with col2:
            st.subheader("Draw box to segment the cells")
            canvas_result = st_canvas(
                fill_color="rgba(255, 229, 204, 0.2)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=background_image if bg_image else None,
                width=background_image.width,
                height=background_image.height,
                # display_toolbar=True,
                drawing_mode=drawing_mode,
                key=f"canvas_{selected_image_name}",  # Unique key for each image
            )

            save_bbox_path = (
                f"{user_folder}/dataset/sam/{selected_image_name}/bbox2.png"
            )

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]

                if len(objects) == 0:
                    st.info("Please draw box to segment the cells.")
                    sys.exit()

                # Initialize an empty list to hold the selected objects
                selected_objects = []

                for obj in objects:
                    # Create a new row with the format [x, y, x+w, y+h]
                    row = [
                        obj["left"],
                        obj["top"],
                        obj["left"] + obj["width"],
                        obj["top"] + obj["height"],
                    ]
                    selected_objects.append(row)

                # Convert the list of selected objects to a numpy array
                selected_objects_new = np.array(selected_objects)
                # st.dataframe(selected_objects_new)

                # Save the NumPy array to a .npy file
                save_npy_path = (
                    f"{user_folder}/dataset/sam/{selected_image_name}/output.npy"
                )
                with st.sidebar:
                    if st.sidebar.button("🤙🏻 Submit"):
                        switch_page("labeling")
                new_mask = predict_from_bbox(
                    selected_image_name + ".png", selected_objects_new[-1].reshape(1, 4)
                )
                annotated_img_path = (
                    f"{user_folder}/dataset/sam/{selected_image_name}/box_annotated.png"
                )
                if os.path.exists(annotated_img_path):
                    pass
                else:
                    annotated_img_path = (
                        f"{user_folder}/dataset/sam/{selected_image_name}/bbox.png"
                    )
                # print("annotated_img_path: ", annotated_img_path)
                # print(np.count_nonzero(new_mask[0]==True))
                annotated_img = apply_colored_masks(
                    annotated_img_path, new_mask[0], ["box"], color_dict
                )
                st.subheader("Real-Time update")

                st.image(annotated_img)

                annotated_img_path = (
                    f"{user_folder}/dataset/sam/{selected_image_name}/box_annotated.png"
                )
                annotated_img.save(annotated_img_path)

                # st.image(annotated_img_path)

            # st.image(annotated_img_path)
            # np.save(save_npy_path, selected_objects_new)

            # return selected_objects_new, selected_image_name

            else:
                st.info("Please draw box to segment the cells.")
                sys.exit()


##########################################################

st.cache_resource.clear()

USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
user_folder = USER_FOLDER_PATH
save_root = SAVE_ROOT_PATH

print(f"Segmentation: user_folder: {USER_FOLDER_PATH}\n")

with st.sidebar:
    login_statement()

segmentation = option_menu(
    "",
    ["✂️ Segmentation with SAM", "🤌 Segmentation by Hand"],
    icons=["-", "-"],
    menu_icon="-",
    default_index=0,
    orientation="horizontal",
    styles=generate_menu_styles(),
)


image_folder = f"{user_folder}/dataset/original/"

if segmentation == "✂️ Segmentation with SAM":
    segmentation_sam(image_folder, save_root, user_folder)
elif segmentation == "🤌 Segmentation by Hand":
    segmentation_hand(user_folder)
