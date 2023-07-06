import time
import streamlit as st
import randfacts
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import os
from streamlit_extras.switch_page_button import switch_page
from css_style import generate_cork_board
from user_util import create_directory_if_not_exists
from page_config import login_statement
from config import USER_FOLDER_PATH, SAVE_ROOT_PATH


def upload_images(image_folder):
    uploaded_files = st.sidebar.file_uploader(
        "Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

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


def display_fact_on_board():
    """Displays a random fact on the board using Streamlit, randfacts, and a custom markdown generator."""

    # This part is assuming that the generate_cork_board function is defined somewhere in your script.
    # from your_module import generate_cork_board

    # Create an empty Streamlit placeholder
    board_placeholder = st.empty()

    # Get a random fact
    fun_info = randfacts.get_fact()

    # Generate the markdown for the cork board with the fact and display it in the placeholder
    board_placeholder.markdown(generate_cork_board(
        fun_info), unsafe_allow_html=True)


##########################################################


# username = handle_username()

# # print(username)
# user_folder = f'./user_folders/{username}'
# create_directory_if_not_exists(user_folder)

# # SAVE_ROOT_PATH = '/Volumes/group05/APP_test/dataset'

# SAVE_ROOT_PATH = f'{user_folder}/dataset/sam'
with st.sidebar:
    login_statement()

user_folder = USER_FOLDER_PATH


image_folder = f'{user_folder}/dataset/original/'
create_directory_if_not_exists(image_folder)
uploaded_image, image_paths = upload_images(image_folder)
# st.write(image_paths)
# st.write(len(image_paths))


if not uploaded_image or not image_paths:
    st.warning("Please upload images")


if uploaded_image:

    with st.expander("Preview of the images"):
        # Divide the width equally among the number of images
        # st.write("preview of the images")

        columns = st.columns(4)  # Create 4 columns
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            with columns[i % 4]:  # Select the column by index
                st.image(image, caption=os.path.basename(
                    image_path), use_column_width=True)
    # columns = st.columns(5)
    if st.sidebar.button("🧀Segmentaiton"):

        display_fact_on_board()

        add_vertical_space(2)
        with st.spinner('Wait for the cell detection...'):

            # style gradient
            gradient = "linear-gradient(to right, #4cd964, #5ac8fa, #007aff, #34aadc, #5856d6, #ff2d55)"

            total_images = len(image_paths)
            progress_placeholders = st.empty()

            for i, image_path in enumerate(image_paths):

                dir = os.path.splitext(os.path.basename(image_path))[0]
                print('dir: ', dir)
                print('image_path: ', image_path)

                os.makedirs(os.path.join(SAVE_ROOT_PATH, dir), exist_ok=True)
                st.write(os.path.basename(image_path))

                from detection import get_image_masks

                get_image_masks(
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
                col1.image(image_path, caption='Orignal image')
                col2.image(f"{SAVE_ROOT_PATH}/{dir}/bbox.png",
                           caption='Segmented image')
                time.sleep(0.1)
            time.sleep(1)

            st.success('Done!', icon="✅")
            # st.experimental_rerun()

        if st.sidebar.button("🤙🏻 Submit"):

            switch_page("Labeling")
