import streamlit as st
import contextlib

@st.cache_data
def generate_cork_board(fun_info):
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    @font-face {{
        font-family: "San Francisco";
        font-weight: 400;
        src: url("https://applesocial.s3.amazonaws.com/assets/styles/fonts/sanfrancisco/sanfranciscodisplay-regular-webfont.woff");
    }}
    div#frame {{
        display: flex; 
        justify-content: center;  
        align-items: center;  
        background-color: #F0F0F0;
        width: 100%;
        height: 250px;
        # padding-top: 35px;
        # padding-left: 35px;
        box-shadow: 0 2px 5px rgba(0,0,0,.15);
    }}

    .note {{
        width: 500px;
        height: 150px;
        box-shadow: 0 3px 6px rgba(0,0,0,.25);
        float: left;
        margin: 0px;
        border: 1px solid rgba(0,0,0,.15);
        background-color: #FFFFFF;
        border-radius: 10px;
    }}

    .text_board {{
        margin: 10px;
        font-family: 'Helvetica Neue', sans-serif;
    }}
    </style>
    """

    cork_board = f"""
    <div id='frame'>
        <div class="note sticky1">
            <div class='text_board'>Do you know: {fun_info}</div>
        </div>
    </div>
    """

    return css + cork_board

def generate_segmentation_progress_bar(count, total_count, gradient, idx):
    display_count = count / total_count * 100
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    @font-face {{
        font-family: "San Francisco";
        font-weight: 400;
        src: url("https://applesocial.s3.amazonaws.com/assets/styles/fonts/sanfrancisco/sanfranciscodisplay-regular-webfont.woff");
    }}
    .flex-container{idx} {{
        display: flex;
        align-items: center;
        margin-bottom: -5px;
        justify-content: flex-start;
        margin-top: 0px
    }}

    .container{idx} {{
        background-color: rgb(192, 192, 192);
        width: 100%;  
        height: 20px; 
        border-radius: 20px;
        margin-bottom: 0px
    }}

    .text{idx} {{
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
        animation: progress-bar-width 1.5s ease-out 1;
        transition: width 1.2s ease-out;
        margin-top: 0px;
        margin-bottom: 0px;
    }}

    .percent{idx} {{
        width: {display_count}%;
    }}

    @keyframes progress-bar-width{idx} {{
        0% {{ width: 0; }}
        100% {{ width: {display_count}%; }}  
    }}
    </style>
    """
    progress_bar = f"""
    <div class="flex-container{idx}">
        <div class="container{idx}">
            <div class="text{idx} percent{idx}">{int(display_count)}%</div> 
        </div>
    </div>
    """
    # Combine CSS and HTML
    custom_progress_bar = css + progress_bar

    return custom_progress_bar

@st.cache_resource(show_spinner=False)
def generate_progress_bar(label,df_count, count, gradient):
    display_count = count/(df_count.max()+1)*100+5
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    @font-face {{
        font-family: "San Francisco";
        font-weight: 400;
        src: url("https://applesocial.s3.amazonaws.com/assets/styles/fonts/sanfrancisco/sanfranciscodisplay-regular-webfont.woff");
    }}

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
        margin: -5px;
        # display: flex;       
        align-items: center;
        font-size: 16px;
        # font-weight: bold;
        text-align: left; 
        color: #333; 
    }}

    .container {{
        background-color: #e3e7eb;
        width: 80%;  
        # flex-grow: 1;
        height: 18px; 
        border-radius: 20px;
        margin-bottom: 0px
    }}

    .text-{label} {{
        background-image: {gradient};
        color: white;
        padding: 0.1%;
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

    progress_bar = f"""
    <div class="flex-container">
        <div class="title">{label}</div>
        <div class="container">
            <div class="text-{label} percent-{label}">{int(count)}</div> 
        </div>
    </div>
    """
    
    # Inject the CSS into the Streamlit app
    st.markdown(css, unsafe_allow_html=True)

    # Create progress bar with custom CSS
    st.markdown(progress_bar, unsafe_allow_html=True)

def generate_count_bar(label, count, total, gradient):
    if total != 0:
        display_count = count / total * 100
    else:
        display_count = 0  

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    @font-face {{
        font-family: "San Francisco";
        font-weight: 400;
        src: url("https://applesocial.s3.amazonaws.com/assets/styles/fonts/sanfrancisco/sanfranciscodisplay-regular-webfont.woff");
    }}

    .vertical-bar-container {{
        height: 300px;
        width: 30px;
        display: flex;
        flex-direction: column-reverse;
        align-items: center;
        background-color: #f2f2f2;
        border-radius: 10px;
        padding: 5px;
        margin-right: 10px;
    }}

    .vertical-bar {{
        background-image: {gradient};
        width: 100%;
        border-radius: 10px;
        transition: height 1s ease-in-out;
    }}

    .percentage {{
        margin-top: 5px;
        font-size: 14px;
        font-weight: bold;
        font-family: "San Francisco", -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .label {{
        font-size: 12px;
        color: #555;
        font-family: "San Francisco", -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    </style>
    """
    count_bar = f"""
    <div class="vertical-bar-container">
        <div class="vertical-bar" style="height: {display_count}%;"></div>
    </div>
    <div class="percentage">{display_count:.1f}%</div>
    <div class="label">{label}</div>
    """
    
    st.markdown(css, unsafe_allow_html=True)

    st.markdown(count_bar, unsafe_allow_html=True)
    
def generate_menu_styles():
    return {

        "container": {
            "padding": "0!important",
            "background-color": '#FFFFFF',
            "text-align": "center",
            "display": "flex",
            "justify-content": "space-around",
            "align-items": "left",
            "list-style-type": "none",
            "margin": "0"
        },
        "menu-title": {
            "color": "#333333",
            "font-weight": "bold"
        },
        "nav-link": {
            "font-size": "15px",
            "text-align": "center",
            "--hover-color": "#90c5ff",
            "color": "#333333",
            "font-weight": "100",
            "text-decoration": "none",
            "transition": "1s",
            "font-weight": "700"
        },
        "nav-link-selected": {
            "background-color": 'white',
            "text-decoration": "underline",
            "font-weight": "700"
        }
    }
    
    
def generate_al_progress_bar(count, total_count, gradient):
    display_count = count / total_count * 100
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
        'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
        sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    @font-face {{
        font-family: "San Francisco";
        font-weight: 400;
        src: url("https://applesocial.s3.amazonaws.com/assets/styles/fonts/sanfrancisco/sanfranciscodisplay-regular-webfont.woff");
    }}
    .flex-container {{
        display: flex;
        align-items: center;
        margin-bottom: -5px;
        justify-content: flex-start;
        margin-top: 0px
    }}

    .container {{
        background-color: rgb(192, 192, 192);
        width: 100%;  
        height: 20px; 
        border-radius: 20px;
        margin-bottom: 0px
    }}

    .text {{
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
        animation: progress-bar-width 1.5s ease-out 1;
        transition: width 1.2s ease-out;
        margin-top: 0px;
        margin-bottom: 0px;
    }}

    .percent {{
        width: {display_count}%;
    }}

    @keyframes progress-bar-width {{
        0% {{ width: 0; }}
        100% {{ width: {display_count}%; }}  
    }}
    </style>
    """
    progress_bar = f"""
    <div class="flex-container">
        <div class="container">
            <div class="text percent">{int(display_count)}%</div> 
        </div>
    </div>
    """
    # Combine CSS and HTML
    custom_progress_bar = css + progress_bar
    st.markdown(custom_progress_bar, unsafe_allow_html=True)

    # return custom_progress_bar


def generate_custom_spinner(text="Loading..."):
    css = """
    <style>
    .heart-rate {
    width: 150px;
    height: 73px;
    position: relative;
    margin: 20px auto;
    }

    .fade-in {
    position: absolute;
    width: 100%;
    height: 100%;
    background-color: #ffffff;
    top: 0;
    right: 0;
    animation: heartRateIn 3s ease-in-out infinite;
    }

    .fade-out {
    position: absolute;
    width: 120%;
    height: 100%;
    top: 0;
    left: -120%;
    animation: heartRateOut 3s ease-in-out infinite;
    background: rgba(255, 255, 255, 1);
    background: -moz-linear-gradient(left, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 1) 50%, rgba(255, 255, 255, 0) 100%);
    background: -webkit-linear-gradient(left, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 1) 50%, rgba(255, 255, 255, 0) 100%);
    background: -o-linear-gradient(left, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 1) 50%, rgba(255, 255, 255, 0) 100%);
    background: -ms-linear-gradient(left, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 1) 50%, rgba(255, 255, 255, 0) 100%);
    background: linear-gradient(to right, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 1) 80%, rgba(255, 255, 255, 0) 100%);
    }

    @keyframes heartRateIn {
    0% {
        width: 100%;
    }
    50% {
        width: 0;
    }
    100% {
        width: 0;
    }
    }

    @keyframes heartRateOut {
    0% {
        left: -120%;
    }
    30% {
        left: -120%;
    }
    100% {
        left: 0;
    }
    }

    .loading-text {
    text-align: center;
    font-size: 14px;
    margin-top: 0px;
    color: #555;
    }
    </style>
    """

    html = f"""<div class="heart-rate">
            <svg version="1.0" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="150px" height="73px" viewBox="0 0 150 73" enable-background="new 0 0 150 73" xml:space="preserve">
                <defs>
                    <linearGradient id="Gradient" x1="0" x2="1" y1="0" y2="0">
                        <stop offset="0%" stop-color="#EECDA3"/>
                        <stop offset="30%" stop-color="#FF69B4"/>
                        <stop offset="80%" stop-color="#EF629F"/>
                        <stop offset="100%" stop-color="#EECDA3"/>
                    </linearGradient>
                </defs>
                <polyline fill="none" stroke="url(#Gradient)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" stroke-miterlimit="10" points="0,45.486 38.514,45.486 44.595,33.324 50.676,45.486 57.771,45.486 62.838,55.622 71.959,9 80.067,63.729 84.122,45.486 97.297,45.486 103.379,40.419 110.473,45.486 150,45.486"
                />
            </svg>
            <div class="fade-in"></div>
            <div class="fade-out"></div>
            </div>
            <div class="loading-text">{text}</div>"""
    # Combine CSS and HTML
    custom_spinner = css + html
    return custom_spinner


@contextlib.contextmanager
def custom_spinner(text='Loading...'):
    spinner_placeholder = st.empty()
    spinner_placeholder.markdown(generate_custom_spinner(text), unsafe_allow_html=True)
    yield
    spinner_placeholder.empty()