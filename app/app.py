import time
import streamlit as st
import randfacts
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import os
from streamlit_extras.switch_page_button import switch_page
from detection import get_image_masks, generate_head2head_comparison

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


def generate_progress_bar(count, total_count, gradient):
    display_count = count / total_count * 100
    css = f"""
    <style>
    .flex-container {{
        display: flex;
        align-items: center;
        margin-bottom: -5px;
        justify-content: flex-start;
        margin-top: 0px
    }}

    .container {{
        background-color: rgb(192, 192, 192);
        width: 600px;  
        height: 20px; 
        border-radius: 20px;
        margin-bottom: 0px
    }}

    .text {{
        background-image: {gradient};
        color: white;
        padding: 0.5%;
        text-align: right;
        font-size: 22px;
        font-weight: bold; 
        text-shadow: 2px 2px 8px #000000; 
        border-radius: 22px;
        height: 100%;
        line-height: 15px;
        width: {display_count}%;  
        animation: progress-bar-width 1.5s ease-out 1;
        transition: width 1.2s ease-out;
        margin-top: 0px;
        margin-bottom: 0px;
    }}

    .percent {{
        width: {display_count}%;
    }}

    @keyframes progress-bar-width {{
        0% {{ width: 0; }}
        100% {{ width: {display_count}%; }}  
    }}
    </style>
    """
    progress_bar = f"""
    <div class="flex-container">
        <div class="container">
            <div class="text percent">{int(display_count)}%</div> 
        </div>
    </div>
    """
    # Combine CSS and HTML
    custom_progress_bar = css + progress_bar

    return custom_progress_bar


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


# def get_image_masks(image_name, save_as_pkl=False, return_elapsed_time=False, return_annotated=False ,para=parameters_segmentation_stack):
#     """
#     returns masks of given image
#     :param image_name: name of the iamge
#     :param save_as_pkl: a flag
#     :param model: default to sam
#     :return: masks, original or as pickled file
#     """
#     st = time.time()
#     sam = sam_model_registry[para['MODEL_TYPE']](checkpoint=para['CHECKPOINT_PATH']).to(device=para['DEVICE'])
#     mask_generator = SamAutomaticMaskGenerator(sam)
#     image_bgr = cv2.imread(image_name)
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     sam_result = mask_generator.generate(image_rgb)
#     masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)][1:]
#     mask_annotator = sv.MaskAnnotator()
#     detections = sv.Detections.from_sam(sam_result=sam_result)
#     annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
#     if save_as_pkl:
#         with open(image_name.removesuffix('.png') + '.pkl', 'wb') as save_segment:
#             pickle.dump(masks, save_segment)
#     et = time.time()
#     duration = et-st
#     print(f'{duration}s')
#     if return_elapsed_time:
#         if return_annotated:
#             return masks, duration, annotated_image
#         else:
#             return masks, duration
#     else:
#         if return_annotated:
#             return masks, annotated_image

##########################################################


image_folder = "image/"
uploaded_image, image_paths = upload_images(image_folder)
# st.write(image_paths)
# st.write(len(image_paths))


if not uploaded_image or not image_paths:
    st.warning("No images uploaded")
else:
    pass

comparison=[]
if uploaded_image:

    if st.sidebar.button("🧀 Segmentaiton"):
        fun_info = randfacts.get_fact()
        st.markdown(generate_cork_board(fun_info), unsafe_allow_html=True)
        add_vertical_space(2)
        with st.spinner('Wait for it...'):
            masks = []
            durations = []
            annotated_images = []
            

            # macOS style gradient
            gradient = "linear-gradient(to right, #4cd964, #5ac8fa, #007aff, #34aadc, #5856d6, #ff2d55)"

            # Create a placeholder for the progress bar
            progress_placeholder = st.empty()

            total_images = len(image_paths)
            for i, image_path in enumerate(image_paths):
                mask, duration, annotated = get_image_masks(
                    image_path, save_as_pkl=True, return_elapsed_time=True, return_annotated=True)
                comparison_image = generate_head2head_comparison(image_path, annotated)
                
                comparison.append(comparison_image)
                masks.append(mask)
                durations.append(duration)
                annotated_images.append(annotated)

                # Calculate the progress as a percentage
                progress = (i + 1) / total_images * 100

                # Update the progress bar
                progress_placeholder.markdown(
                    generate_progress_bar(progress, 100, gradient),
                    unsafe_allow_html=True
                )

                # Sleep for a short time to allow the progress bar to update
                time.sleep(0.1)

            st.success('Done!', icon="✅")
            time.sleep(1)
            st.experimental_rerun()
        
        # st.image(comparison_image, caption='Comparison')

    # Divide the width equally among the number of images

    num_images = len(uploaded_image)
    max_columns = 4
    num_rows = (num_images + max_columns - 1) // max_columns

    for i in range(num_rows):
        columns = st.columns(max_columns)
        for j in range(max_columns):
            image_index = i * max_columns + j
            if image_index < num_images:
                columns[j].image(uploaded_image[image_index])
                




if st.sidebar.button("🤙🏻 Summit"):

    switch_page("Labeling")
