import streamlit as st
from PIL import Image
import altair as alt
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
import pandas as pd
import base64
import glob
import streamlit.components.v1 as components
from PIL import Image
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space


st.set_page_config(
    page_title="AMI05",
    page_icon="👋",
    layout="wide"
)

# function to show image
def img_show(image_index):
    image_path = os.path.join(image_folder, image_files[image_index])
    with open(image_path, "rb") as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    image_html = f'<img src="data:image/png;base64,{image_base64}" alt="Image" \
                    width= "200"; height= "200";>'
    # label_html = f'<p>{get_labels}</p>'
    st.markdown(
        f"""
        <style>
            .custom-container {{
                max-width: 300px;
                height: 200px;
                overflow: auto;
                border: 1px solid transparent;
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: center;
                padding: 10px;
                margin: auto; 
            }}
        </style>
        <div class="custom-container">
            {image_html}
        </div>
        """,
        unsafe_allow_html=True
    )
    # components.html(
    # f"""
    # <style>
    #     .custom-container {{
    #         max-width: 300px;
    #         height: 200px;
    #         overflow: auto;
    #         border: 1px solid transparent;
    #         display: flex;
    #         flex-wrap: wrap;
    #         align-items: center;
    #         justify-content: center;
    #         padding: 10px;
    #         margin: auto; 
    #     }}

    # </style>
    # <div class="custom-container">
    #     {image_html}
    # </div>
    # """,
    # height=200
    # )

def change_amount():
    # if get_labels[image_index] == "RBC":
    #     rbc_count -= 1
    # if get_labels[image_index] == "WBC":
    #     wbc_count -= 1
    # if get_labels[image_index] == "PLT":
    #     plt_count -= 1
    # if get_labels[image_index] == "AGG":
    #     agg_count -= 1
    # if get_labels[image_index] == "OOF":
    #     oof_count -= 1
    # print("!1")
    global option_counts
    option_counts = [len(df_org[df_org['label'] == option]) \
                     - (1 if get_labels[image_index] == option else 0) \
                     + (1 if selected_option == option else 0)
                        for option in options]
    print(option_counts)

def draw_chart(option_counts):
    df = pd.DataFrame({
        'Category': options,
        'Number': option_counts
    })

    colors = alt.Scale(range=["#00EAD3", "#FFF5B7", "#FF449F", "#86A3B8", "#FF8400"])
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Number', title=''),
        y=alt.Y('Category', title=''),
        color=alt.Color('Category', scale=colors, legend=None),
        tooltip=['Number']
    ).properties(
        width=450,
        height=220
    )

    st.markdown("<h1 style='text-align: center;'", unsafe_allow_html=True)
    chart_placeholder.write(chart)
    # st.altair_chart(chart)
# read data from csv

csv_path = 'testapp/cell_input.csv'
df_org = pd.read_csv(csv_path)
get_labels = df_org['label']


df_corr = pd.read_csv('correction_cell.csv')
corr_labels = df_corr['new_label']

rbc_count = len(df_org[df_org['label'] == 'RBC'])
wbc_count = len(df_org[df_org['label'] == 'WBC'])
plt_count = len(df_org[df_org['label'] == 'PLT'])
agg_count = len(df_org[df_org['label'] == 'AGG'])
oof_count = len(df_org[df_org['label'] == 'OOF'])


options = ["RBC", "WBC", "PLT", "AGG", "OOF"]
option_counts = [rbc_count, wbc_count, plt_count, agg_count, oof_count]
# if 'option_counts' not in st.session_state:
#     st.session_state['option_counts'] = [rbc_count, wbc_count, plt_count, agg_count, oof_count]
colors = ["#00EAD3", "#FFF5B7", "#FF449F", "#86A3B8", "#FF8400"]
uncertainties = ["50%", "40%", "30%", "20%", "10%"]     #####
if 'cell_numbers' not in st.session_state:
    st.session_state['cell_numbers'] = []
if 'new_labels' not in st.session_state:
    st.session_state['new_labels'] = []

# start of the page

st.title("Active Learning")

percentage = 60
st.text("Current accuracy of the model:")

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


left_column,  right_column = st.columns([3, 4])

with left_column:
  
    # display one img at one time

    image_folder = "testapp\pages\prediction"
    image_files = os.listdir(image_folder)
    if 'image_index' not in st.session_state:
        st.session_state['image_index'] = 0

    image_index = st.session_state['image_index']

    if image_index >= len(image_files):
        image_index= 0

    # print(image_index)
    
    st.text("Images with maximum uncertainty: " + get_labels[image_index], \
            help= "uncertainty: " + uncertainties[image_index])
    img_show(image_index)

    selected_option = st.selectbox("Correction", options)
   

with right_column:
    # df = pd.DataFrame({
    #     'Category': options,
    #     'Number': option_counts
    # })

    # colors = alt.Scale(range=colors)
    # chart = alt.Chart(df).mark_bar().encode(
    #     x=alt.X('Number', title=''),
    #     y=alt.Y('Category', title=''),
    #     color=alt.Color('Category', scale=colors,  legend=None),
    #     tooltip=['Number']
    # ).properties(
    #     width=450,
    #     height=220
    # )

    # st.markdown("<h1 style='text-align: center;", unsafe_allow_html=True)
    # st.altair_chart(chart)
    chart_placeholder = st.empty()
    # change_amount()
    draw_chart(option_counts)


    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])


    with col1:
        if st.button("Previous"):
            image_index += 1
            print(image_index)
            st.session_state['image_index'] = image_index

    with col2:
        if st.button("Next"):
            image_index += 1
            print(image_index)
            st.session_state['image_index'] = image_index


    with col3:
        if st.button('Save'):
            change_amount()
            chart_placeholder.empty()
            draw_chart(option_counts)
            st.session_state['cell_numbers'].append(image_index)
            st.session_state['new_labels'].append(selected_option)
    
    with col4:
        if st.button('Retrain'):
            data = {
                'cell_number': st.session_state['cell_numbers'],
                'new_label': st.session_state['new_labels']
            }
            df = pd.DataFrame(data)
            df.to_csv('correction_cell.csv', index=False)

    st.markdown(
        """
        <style>
        .stButton>button {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-left: auto;
            margin-right: auto;
            width: 100px;
            height: 45px;
            background-color: #F0F0F0
            margin: 0 auto;
            text-align: center;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    





