import time
import streamlit as st
import randfacts
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import os
from streamlit_extras.switch_page_button import switch_page
from detection import get_image_masks

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)

# parameters_segmentation_stack = {
#     'CHECKPOINT_PATH': os.path.join("app", "sam_weights", "sam_vit_b_01ec64.pth"),
#     'MODEL_TYPE': "vit_b",
#     'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# }


def upload_images(image_folder):
    uploaded_files = st.sidebar.file_uploader(
        "Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    images = []
    image_paths = []  # This will hold the paths of the saved images

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Convert the file to an image
            image = Image.open(uploaded_file)
            # Save the image to the image folder
            image_path = os.path.join(image_folder, uploaded_file.name)
            image.save(image_path)
            images.append(image)

            # Append the path of the saved image to the list
            image_paths.append(image_path)

    return images, image_paths 


def generate_cork_board(fun_info):
    css = f"""
    <style>
    div#frame {{
        background-color: #F0F0F0;
        width: 600px;
        height: 250px;
        padding-top: 35px;
        padding-left: 35px;
        box-shadow: 0 2px 5px rgba(0,0,0,.15);
    }}

    .note {{
        width: 500px;
        height: 160px;
        box-shadow: 0 3px 6px rgba(0,0,0,.25);
        float: left;
        margin: 8px;
        border: 1px solid rgba(0,0,0,.15);
        background-color: #FFFFFF;
        border-radius: 10px;
    }}

    .text_board {{
        margin: 10px;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    </style>
    """

    cork_board = f"""
    <div id='frame'>
        <div class="note sticky1">
            <div class='text_board'>Do you know: {fun_info}</div>
        </div>
    </div>
    """

    return css + cork_board


##########################################################


image_folder = "image/"
uploaded_image, image_paths = upload_images(image_folder)


SAVE_ROOT_PATH = "data"
# st.write(image_paths)
# st.write(len(image_paths))


if not uploaded_image or not image_paths:
    st.warning("Please upload images")
else:
    pass
    # num_images = len(uploaded_image)
    # max_columns = 4
    # num_rows = (num_images + max_columns - 1) // max_columns

    # for i in range(num_rows):
    #     columns = st.columns(max_columns)
    #     for j in range(max_columns):
    #         image_index = i * max_columns + j
    #         if image_index < num_images:
    #             columns[j].image(uploaded_image[image_index])
    #             columns[j].write(f"images{j}")
                
if uploaded_image:
    # Divide the width equally among the number of images
    st.write("preview of the images")
    
    columns = st.columns(4)  # Create 4 columns
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        with columns[i % 4]:  # Select the column by index
            st.image(image, caption=os.path.basename(image_path), use_column_width=True)
    if st.sidebar.button("🧀 Segmentaiton"):
        st.empty()
        fun_info = randfacts.get_fact()
        st.markdown(generate_cork_board(fun_info), unsafe_allow_html=True)
        add_vertical_space(2)
        with st.spinner('Wait for it...'):

            # macOS style gradient
            gradient = "linear-gradient(to right, #4cd964, #5ac8fa, #007aff, #34aadc, #5856d6, #ff2d55)"

            # # Create a placeholder for the progress bar
            # progress_placeholder = st.empty()

            total_images = len(image_paths)
            progress_placeholders = st.empty()

            for i, image_path in enumerate(image_paths):
                
                dir = os.path.splitext(os.path.basename(image_path))[0]
                print('dir: ', dir)
                print('image_path: ', image_path)

                os.makedirs(os.path.join(SAVE_ROOT_PATH, dir), exist_ok=True)

                get_image_masks(
                    i,
                    dir, 
                    image_path, 
                    save_as_pkl=True, 
                    save_annotated=True, 
                    return_elapsed_time=True, 
                    return_annotated=True, 
                    )

                # comparison.append(comparison_image)
                # masks.append(mask)
                # durations.append(duration)

                col1, col2 = st.columns(2)
                col1.image(image_path)
                col2.image(f"{SAVE_ROOT_PATH}/{dir}/bbox.png")
                time.sleep(0.1)
            time.sleep(1)

            # st.success('Done!', icon="✅")
            # st.experimental_rerun()

 


if st.sidebar.button("🤙🏻 Summit"):

    switch_page("Labeling")
