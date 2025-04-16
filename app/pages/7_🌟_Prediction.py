# import pickle
import numpy as np
import sys
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import streamlit.components.v1 as components
from css_style import (
    custom_spinner,
    style_metric,
    display_label_distribution,
    subtitle,
    color_dict,
    generate_legend_html,
    visualize_radar_chart,
)
from PIL import Image
import os
import pandas as pd
from setup.user_util import create_directory_if_not_exists
from setup.config import login_statement, user_folder_config
from labeling.data_loader import (
    load_mask,
    load_data,
    get_or_create_mask_label_df,
)
import shutil
import detection
from streamlit_elements import elements, mui, html, nivo
from feature_calculation import calculate_average_features
from sklearn.preprocessing import MinMaxScaler

from streamlit_echarts import st_echarts


def upload_image(image_folder):
    """
    Allows the user to upload an image file and saves it to a specified folder.

    This function uses Streamlit's file_uploader to allow the user to upload an image file. It then saves this
    image to a specified folder and returns the image and its path.

    Args:
        image_folder (str): The path to the folder where the image should be saved.

    Returns:
        tuple: A tuple containing the uploaded image and the path to the saved image.
    """
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", type=["png", "jpg", "jpeg"]
    )

    image = None
    image_path = None

    if uploaded_file:
        # Convert the file to an image
        image = Image.open(uploaded_file)

        create_directory_if_not_exists(image_folder)

        # Save the image to the image folder
        image_path = os.path.join(image_folder, uploaded_file.name)
        image.save(image_path)

    return image, image_path


@st.cache_data
def apply_colored_masks(image_path, masks, labels, color_dict):
    """
    Apply colored masks to an image.

    Parameters:
    image_path (str): The path to the image file.
    masks (list): A list of masks to apply.
    labels (list): A list of labels corresponding to the masks.
    color_dict (dict): A dictionary mapping labels to colors.

    Returns:
    PIL.Image: The image with the masks applied.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        for idx, mask in enumerate(masks):
            # Check if the label for the current mask is not NaN
            if len(labels) > idx and not pd.isna(labels[idx]):
                # Convert mask to PIL Image and ensure it's 'L' mode
                mask_img = Image.fromarray(
                    np.array(mask).astype("uint8") * 255
                ).convert("L")

                # Get the label for the current mask
                label = labels[idx]

                # Get the color for the current label
                # Default to white if label not in color_dict
                color = color_dict.get(label, (255, 255, 255))

                # Create overlay image
                overlay = Image.new("RGB", img.size, color)

                # Paste the overlay onto the image using the mask
                img.paste(overlay, mask=mask_img)

        return img


@st.cache_data
def process_image(user_folder, image_root, image_name, bbox_path, csv_path, mask_path):
    # st.image(bbox_path)
    print("\nbbox_path: ", bbox_path)
    print("\ncsv_path: ", csv_path)

    if not csv_path:
        st.info("No csv_path")
        sys.exit()

    mask_label_df = get_or_create_mask_label_df(csv_path)
    print("mask_path: ", mask_path)

    box_img = f"{user_folder}/prediction/{image_root}/{image_name}"
    mask_from_pkl = load_mask(mask_path)

    return mask_label_df, box_img, mask_from_pkl


@st.cache_data
def generate_cell_type_report(cell_counts):
    # Title of the report
    st.markdown("## Blood Cell Report")

    # Detailed descriptions
    st.markdown(
        """
    ### Descriptions

    - **Red Blood Cells (RBC)**: They are the most common type of blood cell and the main method of delivering oxygen to body tissues.
    - **Platelets (PLT)**: They are tiny blood cells that help your body form clots to stop bleeding.
    - **White Blood Cells (WBC)**: They are part of the immune system and function in defending the body against both infectious disease and foreign materials.
    - **Aggregate (AGG)**: This refers to the aggregation of several cells from the previous groups. It happens when they are sticking together.
    - **Out of Focus**: This refers to blurry cells that are seen when a cell floats by above or below the focal plane of the microscope.
    """
    )

    # Table of the counts
    st.markdown("### Cell Detection Analysis")

    # Divide the columns into 5 sections
    cols = st.columns(5)

    for cell_type in ["RBC", "PLT", "WBC", "AGG", "OOF"]:
        count = cell_counts.get(cell_type, 0)
        # Default color is #2ECC71

        with cols.pop(0):
            style_metric(label=cell_type, value=count, value_color="#2ECC71")

    if "WBC" in cell_counts:
        st.markdown(
            """
        <p style='font-size:20px; color:red;'>White Blood Cells detected. Please consider further checks for the patient.</p>
        <p>White Blood Cells (WBC) are part of the immune system and are involved in defending the body against infections. The detection of an elevated count of WBCs in the blood may indicate an ongoing infection or inflammation. It is recommended to consult a healthcare professional for a more thorough evaluation and appropriate follow-up.</p>
        """,
            unsafe_allow_html=True,
        )


def create_radar_chart(data, cell_type):
    DATA = []
    row = data.loc[cell_type]
    for col in data.columns:
        DATA.append({"cell": col, "Features": row[col]})
    with elements("nivo_charts"):
        with mui.Box(sx={"height": 400}):
            nivo.Radar(
                data=DATA,
                keys=["Features"],
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
                        "itemHeight": 80,
                        "itemTextColor": "#999",
                        "symbolSize": 10,
                        "symbolShape": "circle",
                        "effects": [
                            {"on": "hover", "style": {"itemTextColor": "#000"}}
                        ],
                    }
                ],
            )
    # st.write("Model's prediction to the cell:")


########################################################


st.cache_resource.clear()

USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
user_folder = USER_FOLDER_PATH
save_root = SAVE_ROOT_PATH
print(f"Prediction: user_folder: {USER_FOLDER_PATH}\n")

with st.sidebar:
    login_statement()
    image_folder = f"{user_folder}/prediction/"
    create_directory_if_not_exists(image_folder)
    uploaded_image, image_path = upload_image(image_folder)

    selected_image = uploaded_image
    print("\nselected_image: ", image_path)

if not image_path:
    st.info("No image uploaded. Please upload an image.")
    sys.exit()
image_name = os.path.basename(image_path)
print("\nimage_name: ", image_name)

image_root, image_extension = os.path.splitext(image_name)
# move the file
target_folder = f"{user_folder}/prediction/{image_root}"
target = f"{user_folder}/prediction/{image_root}/{image_name}"
os.makedirs(target_folder, exist_ok=True)
shutil.copy(image_path, target)

# st.subheader("Original image")
subtitle(f"Original image {image_name}")

st.image(target)

model_folder = f"{USER_FOLDER_PATH}/model/"
#
# List all .joblib files in the model folder
model_files = [file for file in os.listdir(model_folder) if file.endswith(".joblib")]


# Prepend the folder path to each file name
model_paths = {
    os.path.splitext(os.path.basename(file))[0]: os.path.join(model_folder, file)
    for file in model_files
}
# model_paths = sorted(model_paths)
model_name = st.sidebar.selectbox("Model", sorted(list(model_paths.keys())))
# if model_name:
#     st.experimental_rerun()
model_path = model_paths[model_name]
model_name = model_name + ".joblib"
print("\nmodel_name: ", model_name)
print("\nmodel_path: ", model_path)
print("\n")


if st.sidebar.button("Prediction!"):
    with custom_spinner(
        "Wait for the cell detection...",
    ):
        mask_path, csv_path, bbox_path = detection.get_image_masks_with_label(
            user_folder, image_name, model_name
        )

    # test mode:

    # mask_path = f"{user_folder}/prediction/{image_root}/{image_root}.pkl"
    # csv_path = f"{user_folder}/prediction/{image_root}/{image_root}.csv"
    # bbox_path = f"{user_folder}/prediction/{image_root}/{image_root}_bbox.png"

    with st.expander("Hide", expanded=True):
        subtitle("Cell detection")
        st.image(bbox_path)

    mask_label_df, box_img, mask_from_pkl = process_image(
        user_folder, image_root, image_name, bbox_path, csv_path, mask_path
    )

    if (
        "selected_image" not in st.session_state
        or st.session_state["selected_image"] != selected_image
    ):
        # update the selected image in the session state
        st.session_state["selected_image"] = selected_image

        # st.session_state["df"] = load_data(mask_path)
        st.session_state["df"] = load_data(mask_from_pkl)

    # if "points" not in st.session_state:
    #     st.session_state["points"] = []
    if "selected_mask" not in st.session_state:
        st.session_state["selected_mask"] = []

    df = st.session_state["df"]

    if "label" not in df.columns and not df.empty:
        df["label"] = ""

    labels = {"WBC", "RBC", "AGG", "PLT", "OOF"}
    label_lists = {}

    # Check if the 'label' column exists in the DataFrame
    if "label" not in df.columns:
        st.image(box_img)
        st.success("No cells detected.")
        # You can choose to exit the script or perform any other desired action
    else:
        unique_labels = df["label"].unique()

        if len(unique_labels) > 0:
            for label in unique_labels:
                masks = df.loc[df["label"] == label, "masks"].tolist()
                list_name = f"list_masks_{label}"
                label_lists[list_name] = masks

        mask_labels = mask_label_df["label"]
        masks_to_color = df["masks"]

        # col1, col2 = st.columns([4, 1])

        # with col1:
        img = apply_colored_masks(box_img, masks_to_color, mask_labels, color_dict)
        subtitle("Prediction")

        st.image(img)
        html_code = generate_legend_html(color_dict)
        components.html(html_code, width=510, height=70)

        # Count the number of occurrences of each label
        label_counts = mask_label_df["label"].value_counts()

        labels = mask_label_df["label"].unique()
        # desired_order = ["WBC", "RBC", "PLT", "AGG", "OOF"]
        # labels = sorted(labels, key=lambda x: desired_order.index(x))
        # labels = labels.tolist()

        df_feature = calculate_average_features(USER_FOLDER_PATH, image_root)
        df_feature.drop(columns=["cx", "cy"], inplace=True)
        # rename
        df_feature = df_feature.rename(
            columns={
                # 'cx': 'Cx',
                # 'cy': 'Cy',
                "area": "Area",
                "perimeter": "Perimeter",
                "circularity": "Circularity",
                "average_pixel_value": "Average Pixel Value",
                "uniformity": "Uniformity",
            }
        )
        df_feature.index = df_feature.index.str.upper()

        # print(df_feature)
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df_feature),
            columns=df_feature.columns,
            index=df_feature.index,
        )
        print(df_normalized)

        generate_cell_type_report(label_counts)
        display_label_distribution(label_counts, labels, color_dict)

        st.markdown("### Features Analysis")

        st.markdown(
            """
        

          **Feature Description**
        - **Area**: This feature calculates the area of the contour in the image, that is, the total number of pixels within the contour. It can be used to understand the size of the object.


        
        - **Perimeter**: This feature calculates the perimeter of the contour in the image, that is, the length along the contour. It can be used to understand the overall shape size of the object.


        
        - **Circularity**: This feature measures how close the shape of an object is to a circle. The closer the value is to 1, the closer the shape is to a perfect circle.

        
        - **Average Pixel Value**: pixel value This feature calculates the average pixel value of the grayscale image. This can be used to understand the overall brightness or darkness of the image.
        
        - **Uniformity**: This is a measure of the pixel intensity variation within the cell. A high uniformity indicates that the cell's pixel intensities are similar, implying that the cell might have a homogeneous internal structure. On the other hand, low uniformity indicates varying pixel intensities, suggesting a heterogeneous internal structure.
        """
        )

        # st_echarts(option, height="500px")
        # with cols[0]:
        visualize_radar_chart(df_normalized, color_dict)
        cols = st.columns([0.1, 2, 0.1])

        with cols[1]:
            st.dataframe(df_feature)
