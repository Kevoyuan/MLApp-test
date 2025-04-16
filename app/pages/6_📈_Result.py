import streamlit as st
import altair as alt
import pandas as pd
import json
from setup.config import login_statement, user_folder_config
from streamlit_option_menu import option_menu
from css_style import generate_menu_styles
import os
import sys
from sklearn.metrics import auc


def plot_roc(data, al_step):
    """
    Generate an ROC (Receiver Operating Characteristic) curve for the given data.

    Parameters:
    data (dict): The dictionary containing false positive rates, true positive rates, and class labels.
    al_step (int): The active learning step.

    Returns:
    alt.Chart: An Altair chart object representing the ROC curve.
    """
    # Prepare data for ROC curve
    roc_data = []
    for k in data["fpr"][al_step].keys():
        auc_score = auc(data["fpr"][al_step][k], data["tpr"][al_step][k])
        class_name_with_auc = f"{data['int2cells_dict'][k]} (AUC: {auc_score:.2f})"
        for i in range(len(data["fpr"][al_step][k])):
            roc_data.append(
                {
                    "False Positive Rate": data["fpr"][al_step][k][i],
                    "True Positive Rate": data["tpr"][al_step][k][i],
                    "Class": class_name_with_auc.upper(),
                }
            )
    df_roc = pd.DataFrame(roc_data)

    # Plot ROC curve
    roc_chart = (
        alt.Chart(df_roc)
        .mark_line()
        .encode(
            alt.X("False Positive Rate", scale=alt.Scale(zero=False)),
            alt.Y("True Positive Rate", scale=alt.Scale(zero=False)),
            color="Class",
            tooltip=["Class", "False Positive Rate", "True Positive Rate"],
        )
    )

    return roc_chart.properties(title="ROC Curve")


def plot_f1_scores(data):
    """
    Generate a line chart of F1 scores for each class over active learning steps.

    Parameters:
    data (dict): The dictionary containing F1 scores for each class at each active learning step.

    Returns:
    alt.Chart: An Altair chart object representing the F1 scores per class.
    """
    # Prepare data for F1 scores chart
    df_f1 = pd.DataFrame(data["f1_score_class"])
    df_f1["AL step"] = df_f1.index + 1
    df_f1.columns = [
        data["int2cells_dict"].get(str(i), str(i)).upper()
        for i in range(len(df_f1.columns) - 1)
    ] + ["AL step"]
    df_melted = df_f1.melt("AL step", var_name="Class", value_name="F1 Score")

    # Plot F1 scores per class
    line = (
        alt.Chart(df_melted)
        .mark_line()
        .encode(
            alt.X("AL step:O", title="AL step", axis=alt.Axis(labelAngle=0)),
            alt.Y("F1 Score:Q"),
            color="Class:N",  # Treat "Class" as nominal data
        )
    )

    points = (
        alt.Chart(df_melted)
        .mark_point(size=100)
        .encode(
            alt.X("AL step:O", title="AL step", axis=alt.Axis(labelAngle=0)),
            alt.Y("F1 Score:Q"),
            color="Class:N",  # Treat "Class" as nominal data
            tooltip=["AL step", "Class", "F1 Score"],
        )
    )

    return (line + points).properties(title="F1 Scores per Class")


def plot_overall_f1_scores(data):
    """
    Generate a line chart of overall F1 scores over active learning steps.

    Parameters:
    data (dict): The dictionary containing overall F1 scores at each active learning step.

    Returns:
    alt.Chart: An Altair chart object representing the overall F1 scores.
    """
    # Prepare data for overall F1 scores chart
    df_f1_overall = pd.DataFrame(data["f1_score_overall"], columns=["F1 Score"])
    df_f1_overall["AL step"] = df_f1_overall.index + 1

    # Plot overall F1 scores
    line = (
        alt.Chart(df_f1_overall)
        .mark_line()
        .encode(
            alt.X("AL step:O", title="AL step", axis=alt.Axis(labelAngle=0)),
            alt.Y("F1 Score:Q"),
        )
    )

    points = (
        alt.Chart(df_f1_overall)
        .mark_point(size=100)
        .encode(
            alt.X("AL step:O", title="AL step", axis=alt.Axis(labelAngle=0)),
            alt.Y("F1 Score:Q"),
            tooltip=["AL step", "F1 Score"],
        )
    )

    return (line + points).properties(title="Overall F1 Scores")


def plot_accuracy(data):
    """
    Generate a line chart of accuracy scores over active learning steps.

    Parameters:
    data (dict): The dictionary containing accuracy scores at each active learning step.

    Returns:
    alt.Chart: An Altair chart object representing the accuracy scores.
    """
    # Prepare data for accuracy chart
    df_acc = pd.DataFrame(data["acc"], columns=["Accuracy"]) * 100
    df_acc["AL step"] = df_acc.index + 1

    # Plot accuracy
    line = (
        alt.Chart(df_acc)
        .mark_line()
        .encode(
            alt.X("AL step:O", title="AL step", axis=alt.Axis(labelAngle=0)),
            alt.Y("Accuracy:Q"),
        )
    )

    points = (
        alt.Chart(df_acc)
        .mark_point(size=100)
        .encode(
            alt.X("AL step:O", title="AL step", axis=alt.Axis(labelAngle=0)),
            alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=["AL step", "Accuracy"],
        )
    )

    return (line + points).properties(title="Accuracy Scores")


def load_data(filepath):
    """
    Load data from a JSON file.

    Parameters:
    filepath (str): The path to the JSON file.

    Returns:
    dict: The loaded data.
    """
    with open(filepath, "r") as file:
        return json.load(file)


st.cache_resource.clear()
USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
user_folder = USER_FOLDER_PATH
print(f"Result: user_folder: {USER_FOLDER_PATH}\n")
with st.sidebar:
    login_statement()

model_folder = f"{USER_FOLDER_PATH}/results/"

# List all files in the model folder
if os.path.exists(model_folder):
    model_files = os.listdir(model_folder)
else:
    st.info("Model does not exist, please try the Active Learning")
    sys.exit()

# Prepend the folder path to each file name
model_paths = {
    os.path.splitext(os.path.basename(file))[0]: os.path.join(model_folder, file)
    for file in model_files
}


model_name = st.sidebar.selectbox("Model", model_paths)
print("model_path: ", model_name)
data = load_data(f"{USER_FOLDER_PATH}/results/{model_name}.json")
# Show charts

col1, col2, col3 = st.columns([0.1, 1, 1])

result = option_menu(
    "",
    ["AUC", "F1 Score", "Accuracy"],
    icons=[
        "-",
        "-",
        "-",
    ],
    menu_icon="-",
    default_index=0,
    orientation="horizontal",
    styles=generate_menu_styles(),
)

if result == "AUC":
    col1, col2 = st.columns(2)
    with col1:
        if len(data["fpr"]) == 1:
            al_step = 0
        else:
            al_step = st.slider("Select Active Learning step", 1, len(data["fpr"])) - 1
    roc_curve = st.altair_chart(plot_roc(data, al_step), use_container_width=True)

    st.caption(
        "ROC: ROC curve closer to the top-left corner signifies higher accuracy and better ability to distinguish between classes."
    )
    st.caption("AUC: Area under ROC curve, between 0 and 1, the larger the better.")
elif result == "F1 Score":
    f1_scores_curve = st.altair_chart(plot_f1_scores(data), use_container_width=True)
    overall_f1_scores_curve = st.altair_chart(
        plot_overall_f1_scores(data), use_container_width=True
    )
    st.caption(
        "F1 Score: Harmonic mean of precision and recall between 0 and 1, the larger the better."
    )

elif result == "Accuracy":
    accuracy = st.altair_chart(plot_accuracy(data), use_container_width=True)
