import os
import streamlit as st
from requests import get

# @st.cache_data
def handle_username(file_path='username.txt'):
    file_path = 'username'+get_ip_address()+'.txt'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            username = file.read()
    else:
        # st.write("No username set yet.")
        username = None
    return username
 
    
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        
@st.cache_data
def get_directory_size(directory):
    total = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    except NotADirectoryError:
        return os.path.getsize(directory)
    except PermissionError:
        return 0
    return total

def get_ip_address():
    ip = get('https://api.ipify.org').content.decode('utf8')
    # print('My public IP address is: {}'.format(ip))
    return ip

if __name__ == "__main__":
    get_ip_address()