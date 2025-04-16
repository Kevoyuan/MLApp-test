import streamlit as st
from PIL import Image

import pandas as pd
import numpy as np

# import subprocess
import matplotlib.pyplot as plt
import os
import pandas as pd
import base64
import glob

# import streamlit.components.v1 as components
from streamlit_toggle import st_toggle_switch
from PIL import Image, ImageDraw, ImageFont
from streamlit_extras.switch_page_button import switch_page

# from streamlit_extras.add_vertical_space import add_vertical_space
import json
import sys
import cv2
import classification.generic_classifer as generic
from classification.dataset import WbcDataset
from setup.config import login_statement, user_folder_config
from setup.user_util import handle_username
from css_style import display_two_section_progress_bar, generate_menu_styles
import time
import csv
import matplotlib.pyplot as plt
from streamlit_elements import elements, mui, html, nivo
from active_learning_process import (
    retrain_get_accuracy,
    get_uncertainty_and_filename,
    images_moving_and_augmentation,
    read_model,
    get_probability,
    load_test_dataset,
    results_output,
)
import glob
from datetime import timedelta
from datetime import datetime
import joblib


# navigate to the previous cell
def previous():
    st.session_state["rank"] -= 1
    st.experimental_rerun()


# navigate to the next cell
def next():
    st.session_state["rank"] += 1
    st.experimental_rerun()


def filter_data(dataname, col_name1, col_name2, col_value):
    """
    Filters the data based on a given column name and value, and returns the corresponding value from another column.

    Args:
        dataname (pandas.DataFrame): The input DataFrame.
        col_name1 (str): The column name to filter on.
        col_name2 (str): The column name to extract the value from.
        col_value: The value to filter the data on.

    Returns:
        str: The extracted value from col_name2 after filtering the data.

    """
    filtered_data = dataname[dataname[col_name1] == col_value]
    value = filtered_data[col_name2]
    val_str = list(value.reset_index(drop=True))[0]
    return val_str


def filter_files_by_extension(folder_path, extension):
    """
    Filters files in a given folder based on their extension.

    Args:
        folder_path (str): The path to the folder.
        extension (str): The desired file extension.

    Returns:
        list: A list of file names with the specified extension in the folder.

    """
    file_names = []
    for file in glob.glob(os.path.join(folder_path, f"*.{extension}")):
        file_name = os.path.basename(file)
        file_names.append(file_name)
    return file_names


def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def display_classifier_accuracy_table(file_path):
    """
    Displays a table of classifier accuracy based on data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    """
    # Check if file exists
    if not os.path.isfile(file_path):
        st.error("File does not exist: {}".format(file_path))
        sys.exit()

    try:
        # Load the data
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        st.info(
            "Unable to load the file. Please ensure the file is in a valid JSON format. Error: {}".format(
                e
            )
        )
        sys.exit()

    # Flatten the data
    flattened_data = []
    for classifier_name, classifier_data in data.items():
        for data_point in classifier_data["data"]:
            # Convert timestamp to a more readable format
            timestamp = datetime.strptime(data_point["x"], "%Y_%m_%d_%H%M%S")
            formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            flattened_data.append(
                {
                    "Classifier": classifier_name,
                    "Accuracy": data_point["y"],
                    "Date": formatted_timestamp,
                }
            )

    # Convert the flattened data to a pandas DataFrame
    df = pd.DataFrame(flattened_data)

    # Create Styler object
    styler = df.style

    # Apply center alignment to all cells
    styler = styler.set_properties(**{"text-align": "center"})

    df.sort_values(by="Accuracy", ascending=False, inplace=True)

    # Display the DataFrame as a table in Streamlit
    st.dataframe(df, hide_index=True)


def read_acc(file_path, clf, timestamp):
    """
    Reads the accuracy value from a JSON file based on the given classifier and timestamp.

    Args:
        file_path (str): The path to the JSON file.
        clf (str): The classifier name.
        timestamp (str): The timestamp to match.

    Returns:
        str: The accuracy value as a string.

    """
    with open(file_path, "r") as f:
        data = json.load(f)
    clf_data = data[clf]["data"]

    for item in clf_data:
        if item["x"] == timestamp:
            acc = str(item["y"])
            break

    return acc


def index_out_of_range():
    """
    Checks if the current index is out of range based on the number of images in a folder.
    If the index is out of range, it adjusts the index to the valid range.

    """
    if st.session_state["rank"] > len(os.listdir(image_folder)):
        st.session_state["rank"] = len(os.listdir(image_folder))
    if st.session_state["rank"] <= 0:
        st.session_state["rank"] = 1


def is_folder_empty(folder_path):
    return len(os.listdir(folder_path)) == 0


with st.sidebar:
    login_statement()

st.cache_resource.clear()
USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
# username = handle_username()
username = st.session_state["username"]

acc_folder_path = f"{USER_FOLDER_PATH}/accuracy"
# model_folder_path = f"{USER_FOLDER_PATH}/model"

if is_folder_empty(acc_folder_path):
    st.warning("Model does not exist, please try the Classification")
    sys.exit()


css_path = "pages/style.css"
with open(css_path, "r") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

options = ["rbc", "wbc", "plt", "agg", "oof"]
options = [option.upper() for option in options]

if "user_name" not in st.session_state:
    st.session_state["user_name"] = " "

if "start" not in st.session_state:
    st.session_state["start"] = 0

if "model_name" not in st.session_state:
    st.session_state["model_name"] = " "

# st.text(st.session_state['user_name'])
# st.text(username)

if st.session_state["user_name"] != username:
    st.session_state["start"] = 0
    st.session_state["save"] = False
    st.session_state["retrain"] = 0
    st.session_state["user_name"] = username
    st.session_state["rank"] = 1
    st.session_state["selected_option"] = 0
    st.session_state["corrected_option"] = []
    st.session_state["accuracy"] = 0
    st.session_state["file_names"] = []
    st.session_state["data_unlabeled"] = WbcDataset
    st.session_state["AL_time"] = 0
    st.experimental_rerun()


with st.sidebar:
    model_folder = f"{USER_FOLDER_PATH}/model"
    print("\nmodel_folder: ", model_folder)
    # if not os.path.isfile(model_folder):
    #     st.info("Model does not exist, please try the classifier")
    #     sys.exit()
    model_names = filter_files_by_extension(model_folder, "json")
    model_names = sorted(model_names)
    classifier_accuracy_path = f"{USER_FOLDER_PATH}/accuracy/classifier_accuracy.json"
    print("\nclassifier_accuracy_path: ", classifier_accuracy_path)
    # if not os.path.exists(classifier_accuracy_path):
    #     st.info("Please run Classification.")
    #     sys.exit()

    file_names = st.session_state.get("file_names", [])

    if st.session_state["start"] == 0:
        st.session_state["model_name"] = st.selectbox("Model", model_names)
        st.session_state["classifier_string"] = st.session_state["model_name"]
        display_classifier_accuracy_table(classifier_accuracy_path)

        if st.button("start", use_container_width=True):
            st.session_state["start"] = 1
            st.experimental_rerun()

image_folder = f"{USER_FOLDER_PATH}/dataset/train_unlabeled"

image_files = os.listdir(image_folder)

if "save" not in st.session_state:
    st.session_state["save"] = False

if "rank" not in st.session_state:
    st.session_state["rank"] = 1

if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = 0

if "corrected_option" not in st.session_state:
    st.session_state["corrected_option"] = []

if "retrain" not in st.session_state:
    st.session_state["retrain"] = 0

if "accuracy" not in st.session_state:
    st.session_state["accuracy"] = 0

if "file_names" not in st.session_state:
    st.session_state["file_names"] = []

if "data_unlabeled" not in st.session_state:
    st.session_state["data_unlabeled"] = WbcDataset

if "AL_time" not in st.session_state:
    st.session_state["AL_time"] = 0

# Check if the folder is empty
if not os.listdir(image_folder):
    st.info("""No unlabeled cell exist, please click "Finish!" to the result.""")
    st.session_state["start"] = 0
    with st.sidebar:
        if st.button("Finish!", use_container_width=True):
            st.session_state["AL_time"] = 0
            st.balloons()
            switch_page("result")

    st.stop()
    
############################# Get data from classification ################
########## Only run at the beginning or Retrain button clicked ############

if st.session_state["start"] == 0 or st.session_state["retrain"] == 1:
    # uncer file
    data_file_path = f"{USER_FOLDER_PATH}/dataset/data.csv"
    data = [["file name", "uncertainty", "uncertainty rank", "probability"]]

    with open(data_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # label file
    label_file_path = f"{USER_FOLDER_PATH}/dataset/label.csv"
    data_label = [["file name", "label"]]

    with open(label_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data_label)

    # retrain unclick
    if st.session_state["retrain"] == 0:
        st.session_state["rank"] = 1
        classifier, classifier_name = read_model(
            USER_FOLDER_PATH, st.session_state["classifier_string"]
        )

        # accuracy = get_accuracy(USER_FOLDER_PATH, classifier, classifier_name)

        (
            st.session_state["data_unlabeled"],
            uncertainty,
            uncertainty_rank,
            image_names,
        ) = get_uncertainty_and_filename(USER_FOLDER_PATH, classifier, classifier_name)

        uncertainty = np.round(uncertainty, decimals=2)
        uncertainty = [str(value) for value in uncertainty]

        probabilities = get_probability(st.session_state["data_unlabeled"], classifier)
        X_test, y_test, y_test_bin, labels = load_test_dataset(
            USER_FOLDER_PATH, classifier_name
        )
        df = pd.read_csv(data_file_path)
        df["file name"] = image_names
        df["uncertainty"] = uncertainty
        df["uncertainty rank"] = uncertainty_rank
        df.iloc[:, 3] = [",".join(map(str, sublist)) for sublist in probabilities]
        df.to_csv(data_file_path, index=False)

    # retrain click
    if st.session_state["retrain"] == 1:
        st.session_state["retrain"] = 0
        st.session_state["rank"] = 1
        classifier, classifier_name = read_model(
            USER_FOLDER_PATH, st.session_state["classifier_string"]
        )


        # print(image_name_list, label_list)
        
        images_moving_and_augmentation(
            USER_FOLDER_PATH,
            st.session_state["data_unlabeled"],
            st.session_state["file_names"],
            st.session_state["corrected_option"],
            labeled_folder=f"{USER_FOLDER_PATH}/dataset/train_labeled",
        )
        st.session_state["file_names"] = []
        st.session_state["corrected_option"] = []

        classifier, accuracy = retrain_get_accuracy(
            USER_FOLDER_PATH,
            classifier_name,
            st.session_state["classifier_string"],
            classifier,
            st.session_state["AL_time"],
        )
        st.session_state["accuracy"] = int(accuracy)
        if os.listdir(image_folder):
            (
                st.session_state["data_unlabeled"],
                uncertainty,
                uncertainty_rank,
                image_names,
            ) = get_uncertainty_and_filename(USER_FOLDER_PATH, classifier, classifier_name)

            uncertainty = np.round(uncertainty, decimals=2)
            uncertainty = [str(value) for value in uncertainty]
            probabilities = get_probability(st.session_state["data_unlabeled"], classifier)
            df = pd.read_csv(data_file_path)
            df["file name"] = image_names
            df["uncertainty"] = uncertainty
            df["uncertainty rank"] = uncertainty_rank
            df.iloc[:, 3] = [",".join(map(str, sublist)) for sublist in probabilities]
            df.to_csv(data_file_path, index=False)


        model_name_joblib = st.session_state["model_name"].replace(".json", ".joblib")
        joblib.dump(classifier, os.path.join(model_folder, model_name_joblib))
        X_test, y_test, y_test_bin, labels = load_test_dataset(
            USER_FOLDER_PATH, classifier_name
        )
        results_output(
            USER_FOLDER_PATH,
            classifier,
            st.session_state["classifier_string"],
            X_test,
            y_test,
            y_test_bin,
            labels,
            st.session_state["AL_time"],
        )



    df_label = pd.read_csv(label_file_path)
    # df_label["file name"] = image_names
    df_label["file name"] = " "
    df_label["label"] = " "
    df_label.to_csv(label_file_path, index=False)


data_file_path = f"{USER_FOLDER_PATH}/dataset/data.csv"
data = pd.read_csv(data_file_path)

label_file_path = f"{USER_FOLDER_PATH}/dataset/label.csv"
data_label = pd.read_csv(label_file_path)


###############################################################################
# start of the page

# uncer = filter_data('uncertainty rank', 'uncertainty', st.session_state['rank'])

# # Check if the folder is empty
# if not os.listdir(image_folder):
#     st.info("""No unlabeled cell exist, please click "Finish!" to the result.""")
#     st.session_state["start"] = 0
#     with st.sidebar:
#         if st.button("Finish!", use_container_width=True):
#             st.session_state["AL_time"] = 0
#             st.balloons()
#             switch_page("result")

#     st.stop()

try:
    uncer = filter_data(
        data, "uncertainty rank", "uncertainty", st.session_state["rank"]
    )
except IndexError:
    if st.session_state["rank"] <= 0:
        st.warning("Please click ➡️ for more cells")
    elif st.session_state["rank"] > len(os.listdir(image_folder)) and os.listdir(image_folder):
        st.warning(
            """No more cells left in the folder, click "Retrain" for active learning or ⬅️ to check the previous cells"""
        )
    elif not os.listdir(image_folder):
        st.info("""No unlabeled cell exist, please click "Finish!" to the result.""")
        st.session_state["start"] = 0
        with st.sidebar:
            if st.button("Finish!", use_container_width=True):
                st.session_state["AL_time"] = 0
                st.balloons()
                switch_page("result")

        st.stop()


    # sys.exit()
    index_out_of_range()
    uncer = filter_data(
        data, "uncertainty rank", "uncertainty", st.session_state["rank"]
    )


######################## button start unclick ###################################

if st.session_state["start"] == 0:
    with st.sidebar:
        on = st_toggle_switch(
            label="Hyperparameters Preview",
            key="switch_1",
            default_value=False,
            label_after=True,
            inactive_color="#D3D3D3",
            active_color="#FFB266",
            track_color="#FDDFC1",
        )
        if on:
            data_path = os.path.join(model_folder, st.session_state["model_name"])
            hyper_data = read_json_file(data_path)
            st.write(hyper_data)

    st.markdown("### How this page works:")
    st.markdown(
        """- Select the training model from the left and click <span style='background-color: rgb(255,218,185)'>"Start"</span> to begin.""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """- Browse the previous/next image that needs to be labeled by using <span style='background-color: rgb(255,218,185)'>"⬅️/➡️"</span>.""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """- After every labeling, don't forget to <span style='background-color: rgb(255,218,185)'>save</span>!""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """- Click <span style='background-color: rgb(255,218,185)'>"Retrain"</span> to start active lerning progress when you feel the labeled image is enough.""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """- <span style='background-color: rgb(255,218,185)'>"Retrain"</span> can be done several times""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """- Click <span style='background-color: rgb(255,218,185)'>"Finish!"</span> to save the classifier and see the results.""",
        unsafe_allow_html=True,
    )

    if st.session_state["retrain"] == 0:
        parts = st.session_state["model_name"].split("_")
        clf = parts[0]
        timestamp = "_".join(parts[1:]).split(".")[0]
        accuracy = read_acc(classifier_accuracy_path, clf, timestamp)
        st.session_state["accuracy"] = accuracy

######################## button start click ###################################
if st.session_state["start"] == 1:
    # st.markdown('<p style="font-family: Arial; font-size: 20px; color: black;">This is a custom font style</p>', unsafe_allow_html=True)

    col_1, col_2, col_3, col_4 = st.columns([0.1, 4, 0.1, 4])

    with col_2:
        file_name = filter_data(
            data, "uncertainty rank", "file name", st.session_state["rank"]
        )
        col_11, col_22 = st.columns(2)
        with col_11:
            st.write("Uncertainty: " + str(uncer))
        with col_22:
            if file_name in data_label["file name"].values:
                st.write(
                    "Label: "
                    + filter_data(data_label, "file name", "label", file_name).upper()
                )
        placeholder = st.empty()
        # img_show(file_name)
        image_path = os.path.join(image_folder, file_name)
        image_path = Image.open(image_path)
        fig, ax = plt.subplots()
        ax.imshow(image_path)
        st.pyplot(fig)

        print(file_name, st.session_state["rank"])
        selected_option = st.radio("-", options, horizontal=True)
        selected_option = selected_option.lower()

        col1, col2, col3 = st.columns(3)
        with col1:
            # container_html = '''
            # <container class="container1"></container>
            # '''
            # st.markdown(container_html, unsafe_allow_html=True)
            if st.button("←", use_container_width=True, key="left_arrow"):
                st.session_state["save"] = False
                previous()
        with col2:
            if st.button("Save", use_container_width=True):
                file_name = filter_data(
                    data, "uncertainty rank", "file name", st.session_state["rank"]
                )
                st.session_state["file_names"].append(file_name)
                # st.session_state.get("corrected_option").append(selected_option)

                df_label = pd.read_csv(label_file_path)

                # label correction
                if file_name in df_label["file name"].values:
                    row_index = df_label[df_label["file name"] == file_name].index[0]
                    df_label.loc[row_index, "label"] = selected_option
                    df_label.to_csv(label_file_path, index=False)

                else:
                    with open(label_file_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([file_name, selected_option])

                # corrected_option = filter_data(data_label, "file name", "label", file_name)
                # st.session_state["corrected_option"].append(corrected_option)
                next()
                st.experimental_rerun()

        with col3:
            if st.button("→", use_container_width=True):
                st.session_state["save"] = False
                next()

    ##################### chart & general buttons ###########################
    with col_4:
        st.markdown("Current accuracy of the model:")
        # css_style.generate_al_progress_bar(50, 100, 'linear-gradient(to right, red, green)')

        st.markdown(
            f"""

        <div class="container">
            <div class="text percent">{st.session_state['accuracy']}%</div> 
        </div>

        """,
            unsafe_allow_html=True,
        )

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
                background-color: rgb(255, 178, 102);
                color: white;
                padding: 1%;
                text-align: right;
                font-size: 20px;
                border-radius: 20px;
                height: 100%;
                line-height: 15px;
                width: {st.session_state['accuracy']}%;  
                animation: progress-bar-width 1.5s ease-out 1;
            }}
            .percent {{
                width: {st.session_state['accuracy']}%;
            }}
            @keyframes progress-bar-width {{
                0% {{ width: 0; }}
                100% {{ width: {st.session_state['accuracy']}%; }}  
            }}
        </style>
        """

        st.markdown(css, unsafe_allow_html=True)

        st.text(" ")

        with elements("nivo_charts"):
            prob = filter_data(
                data, "uncertainty rank", "probability", st.session_state["rank"]
            )
            substrings = prob.split(",")
            prob_values = [float(substring) for substring in substrings]

            # Streamlit Elements includes 45 dataviz components powered by Nivo.

            DATA = [
                {"cell": "RBC", "Probability": prob_values[1]},
                {"cell": "WBC", "Probability": prob_values[0]},
                {"cell": "PLT", "Probability": prob_values[2]},
                {"cell": "AGG", "Probability": prob_values[3]},
                {"cell": "OOF", "Probability": prob_values[4]},
            ]

            with mui.Box(sx={"height": 280}):
                nivo.Radar(
                    data=DATA,
                    keys=["Probability"],
                    indexBy="cell",
                    valueFormat=">-.2f",
                    margin={"top": 40, "right": 90, "bottom": 60, "left": 80},
                    borderColor={"from": "color"},
                    gridLabelOffset=36,
                    dotSize=5,
                    dotColor={"theme": "background"},
                    dotBorderWidth=2,
                    motionConfig="wobbly",
                    legends=[
                        {
                            "anchor": "top-left",
                            "direction": "column",
                            "translateX": -50,
                            "translateY": -40,
                            "itemWidth": 80,
                            "itemHeight": 20,
                            "itemTextColor": "#999",
                            "symbolSize": 10,
                            "symbolShape": "circle",
                            "effects": [
                                {"on": "hover", "style": {"itemTextColor": "#000"}}
                            ],
                        }
                    ],
                )
            st.write("Model's prediction to the cell:")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Retrain", use_container_width=True):
                st.session_state["AL_time"] += 1
                # st.session_state["file_names"] = data_label
                # print(st.session_state["file_names"],st.session_state["corrected_option"])

                # images_moving(
                #     st.session_state["data_unlabeled"],
                #     image_name_list=st.session_state["file_names"],
                #     label_list=st.session_state["corrected_option"],
                #     labeled_folder=f"{USER_FOLDER_PATH}/dataset/train_labeled",
                # )
                image_name_list = data_label["file name"].tolist()
                label_list = data_label["label"].tolist()
                print(image_name_list)

                st.session_state["file_names"] = image_name_list
                st.session_state["corrected_option"] = label_list

                st.session_state["retrain"] = 1


                st.experimental_rerun()

        with col2:
            if st.button("Finish!", use_container_width=True):
                st.session_state["AL_time"] = 0
                st.session_state["start"] = 0
                st.session_state["save"] = False
                st.balloons()
                switch_page("result")

    with st.expander("How this page works"):
        st.markdown(
            """- Select the training model from the left and click <span style='background-color: rgb(255,218,185)'>"Start"</span> to begin.""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """- Browse the previous/next image that needs to be labeled by using <span style='background-color: rgb(255,218,185)'>"⬅️/➡️"</span>.""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """- After every labeling, don't forget to <span style='background-color: rgb(255,218,185)'>save</span>!""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """- Click <span style='background-color: rgb(255,218,185)'>"Retrain"</span> to start active lerning progress when you feel the labeled image is enough.""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """- <span style='background-color: rgb(255,218,185)'>"Retrain"</span> can be done several times""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """- Click <span style='background-color: rgb(255,218,185)'>"Finish!"</span> to save the classifier and see the results.""",
            unsafe_allow_html=True,
        )

    with st.expander("Example of different cell types"):
        st.image("example_image/example.png")

    with st.sidebar:
        parts = st.session_state["model_name"].split("_")
        clf = parts[0]
        timestamp = "_".join(parts[1:]).split(".")[0]

        acc = read_acc(classifier_accuracy_path, clf, timestamp)
        st.text("Classifier: " + clf)
        st.text("Original accuracy: " + acc + "%")
        # st.text("date: " + timestamp)
