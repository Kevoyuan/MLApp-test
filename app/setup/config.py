# config.py
from setup.user_util import handle_username, create_directory_if_not_exists
from annotated_text import annotated_text
import streamlit as st
import sys
from streamlit_extras.app_logo import add_logo
# import login



def login_statement():
    """
    Handles user login and sets up the Streamlit page.

    This function retrieves the username, checks if it's valid, and sets up the Streamlit page with a custom title and icon.
    It also hides the default Streamlit menu and footer using CSS, and displays a message indicating the logged in user.
    If the user is not logged in, a warning is displayed and the program is terminated.

    """   
    
    # username = handle_username()
    # Access the logged-in user's username
    if "username" in st.session_state:
        username = st.session_state["username"]
    else:
        username = None  # or any default value

    if username == '' or username == None:
        st.warning("Please log in at the main page", icon='🚨')
        sys.exit('Program terminated.')
    st.set_page_config(
        page_title="CellVision",
        page_icon="setup/icon-1024.png",
        layout="wide",
        initial_sidebar_state = 'auto',
    )

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden; }
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    annotated_text(
        "You are logged in as 🕵️ ",
        (f"{username}", '', "#8ef"),)
    add_logo("setup/icon-144.png")
    
    # st.write(f"You are loged in as {username}")

def user_folder_config():
    """
    Configures and returns the paths for the user's folder and dataset.

    This function retrieves the username, creates a user-specific folder if it doesn't exist, and sets up the paths for the user's folder and dataset.

    Returns:
        tuple: A tuple containing the path to the user's folder and the path to the user's dataset.
    """    
    # username = handle_username()
    if "username" in st.session_state:
        username = st.session_state["username"]
    else:
        username = None  # or any default value
    

    # print(username)
    user_folder = f'user_folders/{username}'
    # print(f"Current user folder is: {user_folder}")
    create_directory_if_not_exists(user_folder)

    # SAVE_ROOT_PATH = '/Volumes/group05/APP_test/dataset'
    USER_FOLDER_PATH = user_folder
    SAVE_ROOT_PATH = f'{user_folder}/dataset/sam'
    
    return USER_FOLDER_PATH, SAVE_ROOT_PATH
st.cache_resource.clear()

USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
