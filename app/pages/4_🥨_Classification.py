import streamlit as st
import time
from streamlit_option_menu import option_menu

# from streamlit_toggle import st_toggle_switch
# from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from css_style import generate_menu_styles, custom_spinner
import classification.generic_classifer as generic

# import classification.dataset as dataset
from classification.dataset import WbcDataset
import json
import os
from datetime import datetime
from setup.config import login_statement, user_folder_config
from annotated_text import annotated_text
import sys
from streamlit_extras.colored_header import colored_header
import detection
import random
import shutil

# import numpy as np
import joblib

# from typing import List, Dict
import pandas as pd
from datetime import timedelta
import altair as alt


def get_classifier_params():
    classifier = option_menu(
        "",
        ["KNN", "SVC", "RandomForest", "Logistic", "CNN", "VAE"],
        icons=["-", "-", "-", "-", "-", "-"],
        menu_icon="-",
        default_index=0,
        orientation="horizontal",
        styles=generate_menu_styles(),
    )

    if classifier == "KNN":
        # Depending on the classifier selected, display the appropriate options
        # st.header("KNN")
        colored_header(
            label="KNN",
            description=None,
            color_name="gray-30",
        )

        col1, col2 = st.columns(2)

        with col1:
            n_neighbors = st.slider("n_neighbors", min_value=2, max_value=10, value=5)
            algorithm = st.selectbox(
                "algorithm", ("auto", "ball_tree", "kd_tree", "brute")
            )

        with col2:
            weights = st.selectbox("weights", ("uniform", "distance"))
            metric = st.selectbox("metric", ("l1", "l2"), index=1)

        knn_params = {
            "n_neighbors": n_neighbors,
            "algorithm": algorithm,
            "weights": weights,
            "metric": metric,
        }

    elif classifier == "SVC":
        colored_header(
            label="SVC",
            description=None,
            color_name="gray-30",
        )
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("C", min_value=0.0, max_value=1.0, value=1.0)
            gamma = st.selectbox("gamma", ("scale", "auto"))

        with col2:
            kernel = st.selectbox(
                "kernel", ("rbf", "linear", "sigmoid", "precomputed"), index=0
            )

        col1, col2 = st.columns(2)

        svc_params = {
            "C": C,
            "kernel": kernel,
            "gamma": gamma,
        }

    elif classifier == "RandomForest":
        colored_header(
            label="RandomForest",
            description=None,
            color_name="gray-30",
        )
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider(
                "n_estimators", min_value=1, max_value=200, value=100
            )
            max_depth = st.slider("max_depth", min_value=1, max_value=100, value=None)
        with col2:
            criterion = st.selectbox("criterion", ("gini", "entropy", "log_loss"))
            max_features = st.selectbox("max_features", ("sqrt", "log2", None))

        rf_params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "max_features": max_features,
        }
    elif classifier == "Logistic":
        colored_header(
            label="Logistic",
            description=None,
            color_name="gray-30",
        )
        col1, col2 = st.columns(2)

        with col1:
            penalty = st.selectbox(
                "penalty", ("None", "l1", "l2", "elasticnet"), index=2
            )
            learning_rate = st.selectbox(
                "learning_rate", ("constant", "invscaling", "adaptive")
            )
            C = C = st.slider("C", min_value=0.0, max_value=1.0, value=1.0)

            # alpha = st.number_input("alpha", value=1e-2, format="%.4f")
            # learning_rate_init = st.select_slider(
            #     "learning_rate_init",
            #     options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            # )

        with col2:
            fit_intercept = st.selectbox("fit_intercept", (True, False), index=0)
            class_weight = st.selectbox("class_weight", ("balanced", None), index=1)

            solver = st.selectbox(
                "solver",
                ("lbfgs", "liblinear", "liblinear", "newton-cholesky", "sag", "saga"),
                index=0,
            )

        LR_params = {
            # "hidden_layer_num": hidden_layers,
            "penalty": penalty,
            "C": C,
            "solver": solver,
            "fit_intercept": fit_intercept,
            "class_weight": class_weight,
        }

    # elif classifier == "MLP":
    #     colored_header(
    #         label="MLP",
    #         description=None,
    #         color_name="gray-30",
    #     )
    #     col1, col2 = st.columns(2)

    #     with col1:

    #         solver = st.selectbox("solver", ("lbfgs", "sgd", "adam"), index=2)
    #         learning_rate = st.selectbox(
    #             "learning_rate", ("constant", "invscaling", "adaptive")
    #         )
    #         activation = st.selectbox(
    #             "activation", ("identity", "logistic", "tanh", "relu"), index=3
    #         )
    #         alpha = st.number_input("alpha", value=1e-2, format="%.4f")
    #         learning_rate_init = st.select_slider(
    #             "learning_rate_init",
    #             options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    #         )

    #     with col2:
    #         hidden_layers = st.slider(
    #             "hidden_layer_num", min_value=2, max_value=5, value=2
    #         )

    #         hidden_layer_sizes = []
    #         for i in range(hidden_layers):
    #             size = st.select_slider(
    #                 f"Size of hidden layer {i+1}",
    #                 options=[8, 16, 32, 64, 128],
    #                 value=64,
    #             )
    #             hidden_layer_sizes.append(size)

    #     mlp_params = {
    #         # "hidden_layer_num": hidden_layers,
    #         "hidden_layer_sizes": hidden_layer_sizes,
    #         "activation": activation,
    #         "solver": solver,
    #         "alpha": alpha,
    #         "learning_rate": learning_rate,
    #         "learning_rate_init": learning_rate_init,
    #     }

    elif classifier == "CNN":
        colored_header(
            label="CNN",
            description=None,
            color_name="gray-30",
        )

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.select_slider("batch_size", options=[8, 16, 32, 64])
            max_epoch = st.slider("max_epoch", min_value=20, max_value=100, value=100)
            # activation = st.selectbox(
            #     "activation", ("identity", "logistic", "tanh", "relu"), index=3
            # )
            # alpha = st.number_input('alpha', value=1e-4, format="%.4f")
            learning_rate_init = st.select_slider(
                "learning_rate_init",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            )

        with col2:
            backbone = st.selectbox("backbone", ("vgg11", "resnet18"), index=1)
            solver = st.selectbox("solver", ("lbfgs", "sgd", "adam"), index=2)
            learning_rate = st.selectbox(
                "learning_rate", ("constant", "invscaling", "adaptive"), index=0
            )

        cnn_params = {
            "backbone": backbone,
            "max_epochs": max_epoch,
            "batch_size": batch_size,
            "solver": solver,
            "learning_rate": learning_rate,
            "learning_rate_init": learning_rate_init,
        }
    elif classifier == "VAE":
        colored_header(
            label="VAE",
            description=None,
            color_name="gray-30",
        )

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.select_slider("batch_size", options=[8, 16, 32, 64])
            max_epoch = st.slider("max_epoch", min_value=20, max_value=100, value=100)
            activation = st.selectbox(
                "activation", ("identity", "logistic", "tanh", "relu"), index=3
            )
            alpha = st.number_input("alpha", value=1e-4, format="%.4f")
            learning_rate_init = st.select_slider(
                "learning_rate_init",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            )

        with col2:
            hidden_layers = st.slider(
                "hidden_layer_num", min_value=1, max_value=3, value=2
            )

            hidden_layer_sizes = []
            for i in range(hidden_layers):
                size = st.select_slider(
                    f"Size of hidden layer {i+1}",
                    options=[8, 16, 32, 64, 128],
                    value=64,
                )
                hidden_layer_sizes.append(size)
            solver = st.selectbox("solver", ("lbfgs", "sgd", "adam"), index=2)
            learning_rate = st.selectbox(
                "learning_rate", ("constant", "invscaling", "adaptive"), index=0
            )
        vae_params = {
            "hidden_layer_num": hidden_layers,
            "hidden_layer_sizes": hidden_layer_sizes,
            "alpha": alpha,
            "max_epochs": max_epoch,
            "batch_size": batch_size,
            "activation": activation,
            "solver": solver,
            "learning_rate": learning_rate,
            "learning_rate_init": learning_rate_init,
        }

    if classifier == "KNN":
        selected_classifier_dict = knn_params
    if classifier == "SVC":
        selected_classifier_dict = svc_params
    if classifier == "RandomForest":
        selected_classifier_dict = rf_params
    if classifier == "Logistic":
        selected_classifier_dict = LR_params
    if classifier == "CNN":
        selected_classifier_dict = cnn_params
    if classifier == "VAE":
        selected_classifier_dict = vae_params

    return selected_classifier_dict, classifier


def save_classifier_accuracy(
    classifier_name: str, current_time: str, accuracy: float, file_path: str
):
    # Load existing data
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is not valid JSON, start with an empty dictionary
        data = {}

    # Create a new entry
    new_entry = {"x": current_time, "y": accuracy}

    # If the classifier is already in the data, append the new entry to its list
    if classifier_name in data:
        data[classifier_name]["data"].append(new_entry)
    else:
        # Otherwise, create a new list for this classifier
        data[classifier_name] = {"data": [new_entry]}

    # Save data
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def display_classifier_accuracy_table(file_path):
    # Load the data
    with open(file_path, "r") as f:
        data = json.load(f)

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
                    "Date": formatted_timestamp,
                    "Accuracy": data_point["y"],
                }
            )

    # Convert the flattened data to a pandas DataFrame
    df = pd.DataFrame(flattened_data)

    # Sort the DataFrame by the 'Date' column in descending order (latest date first)
    df.sort_values(by="Date", ascending=False, inplace=True)

    # Display the DataFrame as a table in Streamlit
    st.dataframe(df, hide_index=True)


def move_random_files(source_folder, destination_folder_1, destination_folder_2, n):
    files = os.listdir(source_folder)

    # Randomly select n files
    selected_files = random.sample(files, n)

    for file_name in selected_files:
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder_1, file_name)
        shutil.copyfile(source_file_path, destination_file_path)
        print(f"Moved file: {file_name}")

    # Move the remaining files to another folder
    if destination_folder_2 is None:
        return
    remaining_files = set(files) - set(selected_files)
    for file_name in remaining_files:
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder_2, file_name)
        shutil.copyfile(source_file_path, destination_file_path)
        print(f"Moved remaining file: {file_name}")


def visualize_data(json_file_path):
    # Load data from JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Convert data to DataFrame
    df = pd.DataFrame(
        [
            (model, d["x"], d["y"])
            for model, values in data.items()
            for d in values["data"]
        ],
        columns=["Model", "Date", "Accuracy"],
    )

    # Convert date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y_%m_%d_%H%M%S")

    # Subtract 2 hours from the date values
    df["Date"] -= timedelta(hours=2)

    # Set y-axis range
    y_scale = alt.Scale(domain=[0, 100])

    # x_scale = alt.Scale(domain=(df["Date"].min() , df["Date"].min()))

    x_scale = alt.Scale(
        domain=[
            min(df["Date"]) - timedelta(hours=1),
            max(df["Date"]) + timedelta(hours=1),
        ]
    )

    # Create the scatter plot
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X(
                "Date:T",
                title="Date",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("Accuracy:Q", title="Accuracy", scale=y_scale),
            color=alt.Color("Model:N", legend=alt.Legend(title="Model")),
            tooltip=[
                "Model",
                alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d %H:%M:%S"),
                "Accuracy",
            ],
        )
        .properties(width=600, height=400)
        .interactive()
        # .configure_legend(
        # orient='bottom'
        # )
    )  # Enable zooming and panning

    # Move the legend to the bottom
    # chart = chart.resolve_legend(color='bottom')

    # Display the scatter plot using Streamlit
    # st.title("Accuracy vs. Date")
    st.altair_chart(chart, use_container_width=True)


################################################################
st.cache_resource.clear()
USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
user_folder = USER_FOLDER_PATH
print(f"Classification: user_folder: {USER_FOLDER_PATH}\n")

with st.sidebar:
    login_statement()
selected_classifier_dict, classifier = get_classifier_params()


# Specify the directory path
directory_path = f"{USER_FOLDER_PATH}/accuracy"

# Create the directory if it does not exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

classifier_accuracy_path = f"{USER_FOLDER_PATH}/accuracy/classifier_accuracy.json"

with st.sidebar:
    model_param_path = f"{USER_FOLDER_PATH}/model"
    # on = st_toggle_switch(
    #     label="Hyperparameter",
    #     key="switch_1",
    #     default_value=True,
    #     label_after=True,
    #     inactive_color="#D3D3D3",
    #     active_color="#11567f",
    #     track_color="#29B5E8",
    # )
    # if on:
    # Check if the path exists
    if os.path.exists(model_param_path):
        # Get a list of all json files in the directory
        json_files = [f for f in os.listdir(model_param_path) if f.endswith(".json")]
        # Sort the files based on the first three letters
        json_files = sorted(json_files, key=lambda x: x[:3])
        # Use st.selectbox to select a json file
        selected_model = st.selectbox("My training history", json_files)
        selected_model_param_path = f"{USER_FOLDER_PATH}/model/{selected_model}"
        # Load the JSON file
        with open(selected_model_param_path, "r") as f:
            selected_model_param = json.load(f)
        with st.expander("Hyperparameter: "):
            st.json(selected_model_param)
    else:
        st.write(f"Try the Classifier!")


add_vertical_space(3)
col1, col2, col3 = st.columns([2, 2, 1])


if col2.button("🙈 Pretrain!"):
    detection.save_boxes_from_npy(user_folder)
    # The pretrain function here!
    # Data path
    unlabeled_path = f"{USER_FOLDER_PATH}/dataset/unlabeled"
    labeled_path = f"{USER_FOLDER_PATH}/dataset/labeled"
    test_path = f"{USER_FOLDER_PATH}/dataset/test"
    train_labeled_path = f"{USER_FOLDER_PATH}/dataset/train_labeled"
    train_unlabeled_path = f"{USER_FOLDER_PATH}/dataset/train_unlabeled"
    pregiven_path = f"{USER_FOLDER_PATH}/dataset/pregiven"

    # Move pregiven data to labeled dataset
    if not os.path.exists(pregiven_path):
        shutil.copytree("./prediction", pregiven_path)
        num_pregiven = len(os.listdir(pregiven_path))
        move_random_files(pregiven_path, labeled_path, None, num_pregiven)

    # Calculate the number of test and train data
    num_labeled = len(os.listdir(labeled_path))
    num_unlabeled = len(os.listdir(unlabeled_path))
    num_test = int(0.2 * (num_labeled + num_unlabeled))
    num_train = (num_labeled + num_unlabeled) - num_test
    print(f"num_train: {num_train}, num_test: {num_test}")

    # Create dirs
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)
    if os.path.exists(train_labeled_path):
        shutil.rmtree(train_labeled_path)
    os.makedirs(train_labeled_path)
    if os.path.exists(train_unlabeled_path):
        shutil.rmtree(train_unlabeled_path)
    os.makedirs(train_unlabeled_path)

    # Move data
    move_random_files(labeled_path, test_path, train_labeled_path, int(num_test))
    move_random_files(unlabeled_path, train_unlabeled_path, None, int(num_unlabeled))

    clf = generic.Classifier_cells(classifier, **selected_classifier_dict)
    # Split dataset
    if not os.path.exists(labeled_path):
        st.warning(
            "There is no labeled data, please back to the labeling page and submit your work."
        )
        sys.exit("Program terminated.")
    if classifier == "VAE":
        dataset_train = WbcDataset(
            dir=train_labeled_path,
            split="all",
            transform=None,
            download=False,
            need_label=True,
            resize=True,
        )
        dataset_test = WbcDataset(
            dir=test_path,
            split="all",
            transform=None,
            download=False,
            need_label=True,
            resize=True,
        )
    elif classifier == "CNN":
        dataset_train = WbcDataset(
            dir=train_labeled_path,
            split="all",
            transform=None,
            download=False,
            need_label=True,
            resize=False,
        )
        dataset_test = WbcDataset(
            dir=test_path,
            split="all",
            transform=None,
            download=False,
            need_label=True,
            resize=False,
        )
    else:
        dataset_train = WbcDataset(
            dir=train_labeled_path,
            split="all",
            transform=None,
            download=False,
            need_label=True,
            resize=False,
            need_feature=True,
        )
        dataset_test = WbcDataset(
            dir=test_path,
            split="all",
            transform=None,
            download=False,
            need_label=True,
            resize=False,
            need_feature=True,
        )

    # Training
    X, y = dataset_train.create_X_y()

    with custom_spinner(text="Pretraining..."):
        clf.fit(X, y)
        st.success("Pretrain Done!", icon="✅")

    # Testing
    X, y = dataset_test.create_X_y()
    # Accuracy
    accuracy = int(round(clf.score(X, y), 2) * 100)
    print(f"accuracy: {accuracy}")

    # Save model
    model_dir = f"{USER_FOLDER_PATH}/model/"
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m_%d_%H%M%S", current_time)
    # hyperparameters
    json_file = f"{classifier}_{formatted_time}.json"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    json.dump(selected_classifier_dict, open(f"{model_dir}/{json_file}", "w"))
    # parameters
    model_file = f"{classifier}_{formatted_time}.joblib"
    joblib.dump(clf, os.path.join(model_dir, model_file))

    save_classifier_accuracy(
        classifier, formatted_time, accuracy, classifier_accuracy_path
    )

    annotated_text(
        "Pretrained model accuracy is ",
        (f"{(accuracy)}%", "", "#8ef"),
    )

    # time.sleep(5)
    #
col1, col2 = st.columns([0.7, 1])
with col1:
    try:
        display_classifier_accuracy_table(classifier_accuracy_path)
    except Exception as e:
        pass
with col2:
    try:
        visualize_data(classifier_accuracy_path)
    except Exception as e:
        pass


# if st.sidebar.button("🤙🏻 Submit"):
#     switch_page("activelearning")
