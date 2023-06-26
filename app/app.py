import time
import streamlit as st
import randfacts
from streamlit_extras.add_vertical_space import add_vertical_space
from PIL import Image
import os
from streamlit_extras.switch_page_button import switch_page

def upload_images(image_folder):
    uploaded_files = st.sidebar.file_uploader(
        "Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    images = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Convert the file to an image
            image = Image.open(uploaded_file)
            # Save the image to the image folder
            image_path = os.path.join(image_folder, uploaded_file.name)
            image.save(image_path)
            images.append(image)

        return images
    
    
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


@st.cache_data()
def lottie_local(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

##########################################################


image_folder = "app/image/"
uploaded_image = upload_images(image_folder)
# lottie_url = 'app/cell_animate2.json'
true_time =5



if uploaded_image:
    if st.sidebar.button("🧀 Segmentaiton"):
        fun_info = randfacts.get_fact()
        st.markdown(generate_cork_board(fun_info), unsafe_allow_html=True)
        add_vertical_space(2)

        # st.write("Do you know: "+ fun_info)

        total_time = 100
        # macOS style gradient
        gradient = "linear-gradient(to right, #4cd964, #5ac8fa, #007aff, #34aadc, #5856d6, #ff2d55)"
        
        # Create a placeholder for the progress bar
        progress_placeholder = st.empty()
        for time_elapsed in range(total_time + 1):
            # Update the progress bar in the placeholder
            progress_placeholder.markdown(
                generate_progress_bar(time_elapsed, total_time, gradient),
                unsafe_allow_html=True
            )

            # Sleep for a short time for the first 95%, and a longer time for the last 5%
            if time_elapsed < total_time * 0.95:
                time.sleep(true_time * 0.1/95)  # 1 second for the first 95%
            elif time_elapsed < total_time * 0.97:
                time.sleep(true_time * 0.1/1)  # 1 second for the first 98%
            else:
                time.sleep(true_time * 0.8/3)  # 9 seconds for the last 5%
        # Clear the content of the page
        
        # st.write(total_duration)
        st.success('Done!', icon="✅")
        time.sleep(1)
        st.experimental_rerun()
        

    # Divide the width equally among the number of images


    num_images = len(uploaded_image)
    max_columns = 2
    num_rows = (num_images + max_columns - 1) // max_columns

    for i in range(num_rows):
        columns = st.columns(max_columns)
        for j in range(max_columns):
            image_index = i * max_columns + j
            if image_index < num_images:
                columns[j].image(uploaded_image[image_index])

if st.sidebar.button("🤙🏻 Summit"):
 
    switch_page("Labeling")