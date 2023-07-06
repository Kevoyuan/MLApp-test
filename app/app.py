import streamlit as st
import os
import shutil
from streamlit_toggle import st_toggle_switch
from user_util import get_directory_size
from streamlit_extras.switch_page_button import switch_page
import time

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# function to create a new folder


def create_folder(username):
    folder_path = f'./user_folders/{username}'
    os.makedirs(folder_path, exist_ok=True)
    # This line opens the file for writing, overwriting the file if it exists
    with open('./username.txt', 'w') as file:
        file.write(username)


def main():

    st.header("User Account Management")

    # Input to change username in session state
    username = st.text_input(
        "User Name", key='username')

    menu = ["Login", "SignUp"]
    choice = st.radio('-', menu)

    if choice == "Home":
        st.write("Home")

    elif choice == "Login":
        if st.button("Login"):
            if os.path.isdir(f'./user_folders/{username}'):
                with open('./username.txt', 'w') as file:
                    file.write(username)
                
                st.success(f"Logged in as {username}", icon="✅")
                time.sleep(0.5)
                switch_page("Segmentation")
                
                
            else:
                st.warning("Invalid Username")

    elif choice == "SignUp":
        if st.button("Create Account"):
            if not os.path.isdir(f'./user_folders/{username}'):
                create_folder(username)
                st.success(f"Account created for {username}")
            else:
                st.warning("User already exists")


def delete_user_folder(username):
    folder_path = f'./user_folders/{username}'
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        st.success(f"User {username} deleted successfully.")
    else:
        st.warning("User does not exist.")





if __name__ == '__main__':
    main()
    with st.sidebar:
        on = st_toggle_switch(
            label="Account Management",
            key="switch_1",
            default_value=False,
            label_after=True,
            inactive_color="#D3D3D3",
            active_color="#11567f",
            track_color="#29B5E8",
        )

        if on:
            # List all user folders
            user_folders = [folder for folder in os.listdir(
                './user_folders') if folder != '.DS_Store']

            # Create a Streamlit selectbox for the user folders
            selected_user = st.selectbox("Select a user", user_folders)
            # Calculate and display the size of the selected user's folder
            folder_size = get_directory_size(f'./user_folders/{selected_user}')
            st.write(
                f"The {selected_user} folder is {folder_size / (1024 * 1024):.2f} MB")

            # Create a Streamlit button
            toggle_button = st.checkbox("Delete selected user")

            if toggle_button:
                st.warning(
                    f"Warning: This process will delete all files in the {selected_user} folder!")
                button_clicked = st.button("Confirm")

                # Check if the button is clicked
                if button_clicked:
                    delete_user_folder(selected_user)
