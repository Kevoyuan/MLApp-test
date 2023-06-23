import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page

from streamlit_image_coordinates import streamlit_image_coordinates
import json
from PIL import Image, ImageDraw
import os
import pandas as pd

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)

# Read the CSV file
csv_path = 'testapp/cell_input.csv'
df = pd.read_csv(csv_path, delimiter=',')
cell_labels = df.set_index('cell_number')['label'].to_dict()


# Create separate lists for each label
label_lists = {}
unique_labels = df['label'].unique()

# Iterate over the unique labels and create lists for the first 5 labels only
for i, label in enumerate(unique_labels[:5]):
    coordinates = df.loc[df['label'] == label, 'coordinate_top_left'].apply(
        lambda coord: tuple(map(int, coord.split()))).tolist()
    list_name = f"list_top_left_coordinates_{label}"
    label_lists[list_name] = coordinates

# st.write(label_lists['list_top_left_coordinates_WBC'])


# def is_point_in_box(point, box_top_left, box_size=64):
#     px, py = point
#     bx, by = box_top_left

#     # Calculate the box boundaries
#     x1, y1 = bx, by
#     x2, y2 = bx + box_size, by + box_size

#     # Check if the point is inside the box
#     return x1 <= px <= x2 and y1 <= py <= y2
def is_point_in_box(point, top_left_coordinates, box_size=64):
    px, py = point

    for bx, by in top_left_coordinates:
        # Calculate the box boundaries
        x1, y1 = bx, by
        x2, y2 = bx + box_size, by + box_size

        # Check if the point is inside the box
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True

    return False


def draw_box_if_point_in_box(draw, box, cell_position, points):
    if points:
        current_point = points[-1]
        if is_point_in_box(current_point, cell_position):
            draw.rounded_rectangle(box, radius=10, outline="white", width=2)


def get_box_coordinate(x, y):
    cell_position = (x, y)
    box = (x, y, x + 64, y + 64)
    return cell_position, box


def process_image(image_path, df):
    max_width = 650

    with Image.open(image_path) as img:
        width, height = img.size

        # Calculate the new height based on the desired width and original aspect ratio
        new_height = int((max_width / width) * height)

        # Resize the image using the calculated width and height
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        img = img.convert("RGB")

        draw = ImageDraw.Draw(img)

        if "points" not in st.session_state:
            st.session_state["points"] = []

        # Draw rectangle at each coordinate in points
        for point in st.session_state["points"]:
            # st.write(st.session_state["points"][-1])
            for i, row in df.iterrows():
                x, y = map(int, row['coordinate_top_left'].split())
                cell_position, box = get_box_coordinate(x, y)
                draw_box_if_point_in_box(draw, box, cell_position, st.session_state["points"])

        # Convert the image to RGB mode
        img_rgb = img.convert("RGB")

        value = streamlit_image_coordinates(img_rgb, width=650)

        if value is not None:
            point = value["x"], value["y"]

            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)

                st.experimental_rerun()


highlighted_button_style = """
<style>
    .highlighted {{
        background-color: {0};
        color: white;
        font-size: 20px;
        height: 2em;
        width: 5em;
        border-radius: 10px 10px 10px 10px;
    }}
</style>
"""

normal_button_style = """
<style>
    .normal {{
        background-color: white;
        color: black;
        font-size: 20px;
        height: 2em;
        width: 5em;
        border-radius: 10px 10px 10px 10px;
        border: 2px solid black;
    }}
</style>
"""


def styled_button(col, text, color, label):
    existing_label = cell_labels.get(label)

    if label is None:
        style = normal_button_style.format(color)
        button = col.markdown(
            f'{style}<button class="normal">{text}</button>', unsafe_allow_html=True)

    elif st.session_state["points"] and is_point_in_box(st.session_state["points"][-1], label_lists[f'list_top_left_coordinates_{label}']):
        style = highlighted_button_style.format(color)
        button = col.markdown(
            f'{style}<button class="highlighted">{text}</button>', unsafe_allow_html=True)
    else:
        style = normal_button_style.format(color)
        button = col.markdown(
            f'{style}<button class="normal">{text}</button>', unsafe_allow_html=True)


image_folder = "testapp/image/"
image_files = [f for f in os.listdir(
    image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

col1, col2 = st.columns([5, 1])

# Create a select list with image file options
selected_image = col2.selectbox("Select an image", image_files)

# Construct the full path to the selected image
image_path = os.path.join(image_folder, selected_image)

# Process the image in the left column
with col1:
    process_image(image_path, df)


# # Create buttons for each cell
# buttons = ["WBC", "RBC", "PLT", "AGG", "OOF"]

col1, col2, col3, col4, col5, col6 = st.columns(6)

styled_button(col1, "WBC", "#ff0000", "WBC")
styled_button(col2, "RBC", "#ff9900", "RBC")
styled_button(col3, 'PLT', "#66ccff", 'PLT')
styled_button(col4, "AGG", "#00b300", "AGG")
styled_button(col5, "OOF", "#0066ff", "OOF")
