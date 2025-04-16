import streamlit as st
import os
import shutil
from streamlit_toggle import st_toggle_switch
from setup.user_util import get_directory_size, get_ip_address
from streamlit_extras.switch_page_button import switch_page
import time
import pandas as pd
from streamlit_extras.app_logo import add_logo
from css_style import login_style
from streamlit_option_menu import option_menu
from css_style import generate_menu_styles

# import sys
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

st.set_page_config(
    page_title="CellVision", page_icon="setup/icon-1024.png", layout="wide"
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
add_logo("setup/icon-144.png")

config_path = "config.yaml"
# Load config file
with open(config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)
# st.write(f"config = {config}")
# Create authenticator object
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)
if not hasattr(st.session_state, "username"):
    st.session_state["username"] = None

def create_folder():
    """
    Creates a user-specific folder and writes the username to a file.

    This function creates a folder named after the provided username in the 'user_folders' directory.
    If the folder already exists, no action is taken. The function also writes the username to a file named 'username.txt'.

    Args:
        username (str): The username for which the folder should be created.
    """
    
    username = st.session_state["username"]
    print(f"create foloder username = {username}")
    folder_path = f"./user_folders/{username}"
    os.makedirs(folder_path, exist_ok=True)
    # This line opens the file for writing, overwriting the file if it exists
    file_path = "username" + get_ip_address() + ".txt"
    with open(file_path, "w") as file:
        file.write(f"{username}")


def delete_user_folder(username):
    """
    Deletes a user-specific folder.

    This function deletes a folder named after the provided username in the 'user_folders' directory.

    Args:
        username (str): The username for which the folder should be deleted.
    """
    folder_path = f"./user_folders/{username}"
    shutil.rmtree(folder_path)


def login_module():
    login_style()
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image("setup/cellvision4.png")
            st.divider()

            # Select operation from option menu
            operation = option_menu(
                "",
                [
                    "Login",
                    "Reset Password",
                    "Register User",
                    # "Forgot Password",
                    # "Forgot Username",
                ],
                icons=["-", "-", "-"],
                menu_icon="-",
                default_index=0,
                orientation="horizontal",
                styles=generate_menu_styles(),
            )
            if operation == "Login":
                # Render login widget
                name, authentication_status, username = authenticator.login('Login', 'main')
                

                if authentication_status:
                    authenticator.logout('Logout', 'main', key='unique_key')
                    st.success(f'Welcome 🎉 *{name}*', icon="✅")
                    # st.title('Some content')
                    # Store logged-in user's username in session state
                    st.session_state["username"] = username
                    print(f"username = {username}")
                    create_folder()
                    
                    # time.sleep(1.5)
                    # switch_page("Segmentation")
                elif authentication_status is False:
                    st.error('Username/password is incorrect')
                elif authentication_status is None:
                    st.warning('Please enter your username and password')

            elif operation == "Reset Password":
                # Allow user to reset password
                if st.session_state["username"]:
                    try:
                        if authenticator.reset_password(st.session_state["username"], 'Reset password'):
                            st.success('Password modified successfully')
                            # Save updated configuration
                            with open(config_path, 'w') as file:
                                yaml.dump(config, file, default_flow_style=False)
                    except Exception as e:
                        st.error(e)

            elif operation == "Register User":
                # Allow user to register
                try:
                    username= authenticator.register_user('Register user', preauthorization=False)
                    # st.session_state["username"] = username
                    
                    if username:
                        st.success('You registered successfully', icon="✅")
                        
                        
                        # Save updated configuration
                        with open(config_path, 'w') as file:
                            yaml.dump(config, file, default_flow_style=False)
                        st.experimental_rerun()
                except Exception as e:
                    st.error(e)

            elif operation == "Forgot Password":
                # Handle forgotten password
                try:
                    username_of_forgotten_password, email_of_forgotten_password, new_random_password = authenticator.forgot_password('Forgot password')
                    if username_of_forgotten_password:
                        st.success('New password sent securely')
                        # Remember to send new_random_password to user securely
                    else:
                        st.error('Username not found')
                except Exception as e:
                    st.error(e)

            # elif operation == "Forgot Username":
            #     # Handle forgotten username
            #     try:
            #         username_of_forgotten_username, email_of_forgotten_username = authenticator.forgot_username('Forgot username')
            #         if username_of_forgotten_username:
            #             st.success('Username sent securely')
            #             # Remember to send username_of_forgotten_username to user securely
            #         else:
            #             st.error('Email not found')
            #     except Exception as e:
            #         st.error(e)


def manage_user_folders():
    """
    Manages user folders in a Streamlit app.

    This function lists all user folders, calculates their sizes, and displays this information in a table.
    It also provides an interface for selecting a user and deleting their folder.
    """
    # Define the path to user folders
    user_folders_path = "./user_folders"

    if "username" in st.session_state:
        username = st.session_state["username"]
    else:
        username = None  # or any default value
    # Check if the path exists
    if not os.path.exists(user_folders_path):
        print(f"The path {user_folders_path} does not exist.")
        st.warning("Please Login/SignUp!")
        return

    # List all user folders excluding the ".DS_Store" and "None"
    user_folders = [
        folder
        for folder in os.listdir(user_folders_path)
        if folder not in [".DS_Store", "None"]
    ]

    # Calculate the maximum folder size
    if os.path.exists(user_folders_path):
        MAX_FOLDER_SIZE = max(
            get_directory_size(os.path.join(user_folders_path, user))
            for user in user_folders
        )

    # Create a list to store the user data
    user_data = []

    # Calculate the size of each user's folder and add it to the list
    for user in user_folders:
        folder_size = get_directory_size(os.path.join(user_folders_path, user))
        user_data.append([user, folder_size / (1024 * 1024)])

    # Convert the list to a DataFrame
    df = pd.DataFrame(user_data, columns=["User", "Size (MB)"])

    # Round the Size (MB) column to 2 decimal places
    df["Size (MB)"] = df["Size (MB)"].round(2)

    # Sort the DataFrame by folder size in descending order
    df = df.sort_values(by="Size (MB)", ascending=False)

    # Reset the index of the DataFrame and make the index start from 1 instead of 0
    df.reset_index(drop=True, inplace=True)
    df.index += 1

    # Display the DataFrame
    st.dataframe(df)

    # # Select a user
    # selected_user = st.selectbox("Select a user", user_folders)

    # # Delete the selected user's folder
    # if st.checkbox(f"Delete {selected_user}"):
    #     st.warning(
    #         f"Warning: This process will delete all files in the {selected_user} folder!"
    #     )
    #     if st.button("Confirm"):
    #         remove_user_credentials(selected_user)
    #         # delete_user_folder(selected_user)
    #         st.experimental_rerun()

def remove_user_credentials(username):
    """
    Removes a user's credentials.

    This function removes the credentials for the provided username from the 'config' dictionary and writes the updated dictionary back to the YAML file.

    Args:
        username (str): The username for which the credentials should be removed.
    """
    # Load the YAML file
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Check if the username exists in the credentials
    if username in config["credentials"]:
        # Remove the user's credentials
        del config["credentials"][username]

        # Write the updated config back to the YAML file
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        st.success(f"Credentials for {username} removed successfully.")
    else:
        st.error(f"No credentials found for {username}.")


def main():
    login_module()


if __name__ == "__main__":
    st.cache_resource.clear()
    main()
    # Sidebar for account management
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
            manage_user_folders()
            # except
            # st.warning("Please Login/SignUp")
