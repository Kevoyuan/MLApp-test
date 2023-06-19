
import pickle
import numpy as np
from streamlit_toggle import st_toggle_switch
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox
import time
from streamlit_image_coordinates import streamlit_image_coordinates

from PIL import Image, ImageDraw
import os
import pandas as pd

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)

def upload_image(image_path):
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        # Save the image to the image folder
        image_path = os.path.join(image_folder, uploaded_file.name)
        image.save(image_path)

        return image

    return None


def generate_time_progress_bar(label, elapsed_time, total_time, gradient):
    # Calculate the progress percentage
    display_count = elapsed_time / total_time * 100 if total_time != 0 else 0

    css = f"""
    <style>
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
        margin-bottom: -5px;
        align-items: center;
        font-size: 18px;
        text-align: left; 
    }}

    .container {{
        background-color: rgb(192, 192, 192);
        width: 80%;  
        height: 20px; 
        border-radius: 20px;
        margin-bottom: 0px
    }}

    .text-{label} {{
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
            <div class="text-{label} percent-{label}">{int(elapsed_time)} / {int(total_time)}</div> 
        </div>
    </div>
    """, unsafe_allow_html=True)
    
def progress_bar(percent: int):
    return f"""
    <div style="width: 100%; background-color: #f3f3f3; border-radius: 10px; padding: 3px;">
        <div style="width: {percent}%; height: 20px; background-color: #1e88e5; border-radius: 10px;"></div>
    </div>
    """    

##########################################################

image_folder = "testapp/image/"
uploaded_image = upload_image(image_folder)


if st.sidebar.button("🧀 Segmentaiton"):
    pass

elapsed_time=30
total_time=100
# Call the function with elapsed time, total time, label and gradient

def generate_time_progress_bar(label, total_time, gradient):
    css = f"""
    <style>
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
        margin-bottom: -5px;
        align-items: center;
        font-size: 18px;
        text-align: left; 
    }}

    .container {{
        background-color: rgb(192, 192, 192);
        width: 80%;  
        height: 20px; 
        border-radius: 20px;
        margin-bottom: 0px
    }}

    .text-{label} {{
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
        width: 0%;  
        animation: progress-bar-width-{label} 1.5s ease-out 1;
        transition: width 1.2s ease-out;
        margin-top: 0px;
        margin-bottom: 0px;
    }}

    .percent-{label} {{
        width: 0%;
    }}

    @keyframes progress-bar-width-{label} {{
        0% {{ width: 0; }}
        100% {{ width: 100%; }}  
    }}
    </style>
    """

    # Inject the CSS into the Streamlit app
    st.markdown(css, unsafe_allow_html=True)

    for elapsed_time in range(total_time + 1):
        # Calculate the progress percentage
        display_count = elapsed_time / total_time * 100 if total_time != 0 else 0

        # Create progress bar with custom CSS
        st.markdown(f"""
        <div class="flex-container">
            <div class="title">{label}</div>
            <div class="container">
                <div class="text-{label}" style="width: {display_count}%;">{elapsed_time} / {total_time}</div> 
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Update progress bar every second
        time.sleep(1)
        
        
generate_time_progress_bar(
    label='Progress', 
    total_time=100, 
    gradient="linear-gradient(to right, #ff0000, #ffff00)"
)