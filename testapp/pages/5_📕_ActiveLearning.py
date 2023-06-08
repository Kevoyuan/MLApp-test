import streamlit as st
from PIL import Image
import altair as alt
import pandas as pd
import subprocess
import os
import base64
import glob
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space



st.set_page_config(
    page_title="AMI05",
    page_icon="👋",
    layout="wide"
)

st.title("Active Learning")

percentage = 60
st.write("Current accuracy of the model:")

st.markdown(f"""

<div class="container">
    <div class="text percent">{percentage}%</div> 
</div>

""", unsafe_allow_html=True)

css = f"""
<style>
    p {{
        font-size: 18px;
    }}
    .container {{
        background-color: rgb(192, 192, 192);
        width: 100%;
        height: 30px;
        margin: auto;
        border-radius: 20px;
    }}
    .text {{
        background-color: rgb(116, 194, 92);
        color: white;
        padding: 1%;
        text-align: right;
        font-size: 20px;
        border-radius: 20px;
        height: 100%;
        line-height: 15px;
        width: {percentage}%;  
        animation: progress-bar-width 1.5s ease-out 1;
    }}
    .percent {{
        width: {percentage}%;
    }}
    @keyframes progress-bar-width {{
        0% {{ width: 0; }}
        100% {{ width: {percentage}%; }}  
    }}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# load img

st.write("Images need to be labeled")

image_folder = "Z:/prediction"

image_files = glob.glob(image_folder + "/*.png")

image_tags = ""

for image_file in image_files:
    with open(image_file, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    image_tags += f'<img src="data:image/png;base64,{image_base64}" alt="Image">'

left_column, right_column = st.columns([3, 1])
with left_column:
    components.html(
        f"""
        <style>
            .custom-container {{
                max-width: 500px;
                height: 500px;
                overflow: auto;
                border: 1px solid rgba(0, 0, 0, 0.2);
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: center;
                padding: 10px;
                margin: auto; 
            }}
        </style>
        <div class="custom-container">
            {image_tags}
        </div>
        """,
        height=550
    )

with right_column:
    options = ["rbc", "wbc", "plt", "agg", "oof"]
    selected_option = st.selectbox("Label the cell", options)
    st.write("Label:", selected_option)

    

col1, col2 = st.columns(2)

# with col1:
#     # if st.button("Labeling"):
#     options = ["rbc", "wbc", "plt", "agg", "oof"]
#     selected_option = st.selectbox("Label the cell", options)
#     st.write("Label:", selected_option)
    

with col1:
    # if st.button("Retraining"):
    st.button("Save")

with col2:
    st.button("Retrain")
    # if st.button("Next"):
    #     st.session_state.current_image_index += 1
    #     if st.session_state.current_image_index >= len(image_files):
    #         st.session_state.current_image_index = 0
    #     current_image.image(image_files[st.session_state.current_image_index])


st.markdown(
    """
    <style>
    .stButton>button {
        display: block;
        margin: 0 auto;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# if st.button("Labeling"):
#     subprocess.run(["python", "testapp/pages/1_🍰_Segmentation.py"])

# if st.button("Retraining"):
#     ###### to be continue.. #####
#     st.write("11")
