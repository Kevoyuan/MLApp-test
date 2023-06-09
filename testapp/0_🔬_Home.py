from streamlit_toggle import st_toggle_switch
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox

from streamlit_image_coordinates import streamlit_image_coordinates

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
df_original = pd.read_csv(csv_path, delimiter=',')
df = df_original.copy()

labels = {'WBC', 'RBC', 'AGG', 'PLT', 'OOF'}

# Create separate lists for each label
label_lists = {}

# Iterate over the labels and create lists for each
for label in labels:
    coordinates = df.loc[df['label'] == label, 'coordinate_top_left'].apply(
        lambda coord: tuple(map(int, coord.split()))).tolist()
    list_name = f"list_top_left_coordinates_{label}"
    label_lists[list_name] = coordinates

# st.write(label_lists['list_top_left_coordinates_WBC'])
def upload_image(image_path):
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        # st.image(image, caption='Uploaded Image', use_column_width=True)
        # Save the image to the image folder
        image_path = os.path.join(image_folder, uploaded_file.name)
        image.save(image_path)


        return image

    return None

def delete_file(file_path):
    """Delete a file at the given path."""
    if os.path.isfile(file_path):
        os.remove(file_path)
        st.success("File removed successfully.")
    else:
        st.error("File not found.")
        
def delete_image(image_folder):
    """Allow users to delete an image from the image folder."""

    # List all image files in the directory
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

    # Let the user select an image to delete
    selected_image = selectbox("Select an image to delete", image_files)

    if selected_image is not None:
        # Delete the selected image
        image_path = os.path.join(image_folder, selected_image)
        if st.sidebar.button('Delete Image'):
            delete_file(image_path)


def is_point_in_box(point, top_left_coordinates, box_size=96):
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
            # Draw a highlighted box
            draw.rectangle(box, outline="red", width=2)


def change_label(point, new_label, df, csv_path='testapp/cell_input.csv'):
    """Change the label of the cell at the given point to the new label and save the changes to a CSV file."""
    for i, row in df.iterrows():
        x, y = map(int, row['coordinate_top_left'].split())
        cell_position = (x, y)
        box_size = 96
        x1, y1 = cell_position
        x2, y2 = x1 + box_size, y1 + box_size
        px, py = point
        if x1 <= px <= x2 and y1 <= py <= y2:
            df.loc[i, 'label'] = new_label
            # Save the DataFrame back to the CSV file
            df.to_csv(csv_path, index=False)
            return


def styled_button(col, text, label, df):
    """Create a styled button. When the button is clicked, change the label of the highlighted cell to the button's label."""
    # If the current button label matches the last clicked button label, turn it green
    if st.session_state.get("last_clicked_button") == label:
        button_color = '💘'  # green
    # If the current button's label matches the label of the point in the image that was last clicked, turn it green
    elif label is not None and st.session_state["points"] and is_point_in_box(st.session_state["points"][-1], label_lists[f'list_top_left_coordinates_{label}']):
        button_color = '💘'  # green
    else:
        button_color = '🤍'  # white

    # Create the button
    button_clicked = col.button(f'{button_color} {text}')

    # If the button is clicked, change the label of the highlighted cell to this button's label and update the last clicked button
    if button_clicked and st.session_state["points"]:
        st.session_state['last_clicked_button'] = label
        change_label(st.session_state["points"][-1], label, df)
        st.experimental_rerun()
    # If the button is not clicked but a point in the image is clicked, reset the last clicked button
    elif not button_clicked and st.session_state["points"]:
        st.session_state['last_clicked_button'] = None


def generate_progress_bar(label, count, gradient):
    display_count = count/(data['Count'].max()+1)*100+10
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
        # display: flex;       
        align-items: center;

        
        font-size: 18px;
        # font-weight: bold;
        text-align: left; 
    }}

    .container {{
        background-color: rgb(192, 192, 192);
        width: 80%;  
        # flex-grow: 1;
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
            <div class="text-{label} percent-{label}">{int(count)}</div> 
        </div>
    </div>
    """, unsafe_allow_html=True)


def clear_column(csv_path, column_name):
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Clear the values in the specified column
    df[column_name] = ""

    # Save the modified DataFrame back to the CSV file
    df.to_csv(csv_path, index=False)

###############################################################


image_folder = "testapp/image/"
uploaded_image = upload_image(image_folder)

image_files = [f for f in os.listdir(
    image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

# col1, col2 = st.columns([5, 1])

# Create a select list with image file options
selected_image = st.sidebar.selectbox("Select an image", image_files)

# Construct the full path to the selected image
image_path = os.path.join(image_folder, selected_image)

if "points" not in st.session_state:
    st.session_state["points"] = []

# with col1:
value = streamlit_image_coordinates(image_path,width=650)


if value is not None:
    point = value["x"], value["y"]

    # Check if the point is inside any labeled box
    is_in_box = any(is_point_in_box(
        point, label_lists[f'list_top_left_coordinates_{label}']) for label in labels)

    if not is_in_box:
        # If the point is not inside any labeled box, set the last clicked button to None
        st.session_state['last_clicked_button'] = None

    if point not in st.session_state["points"]:
        st.session_state["points"].append(point)

        st.experimental_rerun()


# # Create buttons for each cell
col1, col2, col3, col4, col5 = st.columns(5)


styled_button(col1, "WBC", "WBC", df)
styled_button(col2, "RBC", "RBC", df)
styled_button(col3, 'PLT', 'PLT', df)
styled_button(col4, "AGG", "AGG", df)
styled_button(col5, "OOF", "OOF", df)


label_counts = df['label'].value_counts()
complete_label_counts = pd.Series(0, index=labels)
complete_label_counts = complete_label_counts.add(label_counts, fill_value=0)

# st.write(complete_label_counts)

with st.sidebar:
    on = st_toggle_switch(
        label="Advance Setting",
        key="switch_1",
        default_value=False,
        label_after=True,
        inactive_color="#D3D3D3",
        active_color="#11567f",
        track_color="#29B5E8",

    )
    if on:
        # Create a Streamlit button
        toggle_button = st.sidebar.checkbox("Clear labels")
        
        if toggle_button:
            st.sidebar.warning("Warning: This process will clear the labels you have marked!")
            button_clicked = st.sidebar.button("Confirm")
            

            # Check if the button is clicked
            if button_clicked:
                clear_column(csv_path, 'label')
                st.sidebar.success("The cell label has been cleared.")
        delete_image(image_folder)


###########################################################################

# Bar chart

# initial data
data = complete_label_counts.reset_index()
data.columns = ['Label', 'Count']


wbc_count = data.loc[data['Label'] == 'WBC', 'Count'].iloc[0]
rbc_count = data.loc[data['Label'] == 'RBC', 'Count'].iloc[0]
plt_count = data.loc[data['Label'] == 'PLT', 'Count'].iloc[0]
agg_count = data.loc[data['Label'] == 'AGG', 'Count'].iloc[0]
oof_count = data.loc[data['Label'] == 'OOF', 'Count'].iloc[0]


generate_progress_bar(
    "WBC", wbc_count, "linear-gradient(to right, #d4fc79 0%, #96e6a1 100%)")  # Dusty Grass
generate_progress_bar(
    'RBC', rbc_count, "linear-gradient(to right, #DECBA4, #3E5151)")  # Sand to Blue
generate_progress_bar(
    'PLT', plt_count, "linear-gradient(to right, #8360c3, #2ebf91)")  # Kye Meh
generate_progress_bar(
    'AGG', agg_count, "linear-gradient(to right, #8e2de2, #4a00e0)")  # Amin
generate_progress_bar(
    'OOF', oof_count, "linear-gradient(to right, #fffbd5, #b20a2c)")  # Relaxing Red


###############################################################


add_vertical_space(2)

# Create three columns
# col1, col2, col3 = st.columns([2, 2, 1])
if st.sidebar.button("🤙🏻 Summit"):
    switch_page("Classification")
