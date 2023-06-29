import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_toggle import st_toggle_switch
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)
classifier = option_menu('Classification Model', ['KNN', 'SVC', 'Random Forest', 'MLP', 'VAE'],
                         icons=['-', '-', '-', '-', '-'],
                         menu_icon="bi bi-diagram-3-fill", default_index=0, orientation="horizontal",
                         styles={
    "container": {
        "padding": "0!important",
        "background-color": '#FFFFFF',  # lighter grey color for macOS-like appearance
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
        # "margin":"30px",
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
        "font-weight": "700",
        # "color": "white",

    }
}
)


if classifier == 'KNN':

    # Depending on the classifier selected, display the appropriate options
    st.header("KNN")

    col1, col2 = st.columns(2)

    with col1:
        n_neighbors = st.slider(
            'n_neighbors', min_value=2, max_value=10, value=5)
        algorithm = st.selectbox(
            'algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))

    with col2:
        weights = st.selectbox('weights', ('uniform', 'distance'))
        metric = st.selectbox('metric', ('l1', 'l2'))

    knn_params = {
        "n_neighbors": n_neighbors,
        "algorithm": algorithm,
        "weights": weights,
        "metric": metric,
    }


elif classifier == "SVC":
    st.header("SVC")
    col1, col2 = st.columns(2)
    with col1:
        C = st.slider('C', min_value=0.0, max_value=1.0, value=1.0)
        gamma = st.selectbox('gamma', ('scale', 'auto'))

    with col2:
        kernel = st.selectbox(
            'kernel', ('linear', 'rbf', 'sigmoid', 'precomputed'))

    col1, col2 = st.columns(2)

    svc_params = {
        'C': C,
        'kernel': kernel,
        'gamma': gamma,
    }

elif classifier == "Random Forest":
    st.header("Random Forest")
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider(
            'n_estimators', min_value=1, max_value=200, value=100)
        max_depth = st.slider('max_depth', min_value=1,
                              max_value=100, value=10)
    with col2:
        criterion = st.selectbox('criterion', ('gini', 'entropy', 'log_loss'))
        max_features = st.selectbox('max_features', ('sqrt', 'log2', None))

    rf_params = {
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_depth': max_depth,
        'max_features': max_features
    }

elif classifier == "MLP":
    st.header("MLP")
    col1, col2 = st.columns(2)

    with col1:
        hidden_layer_sizes = st.slider(
            'hidden_layer_sizes', min_value=1, max_value=100, value=50)
        solver = st.selectbox('solver', ('lbfgs', 'sgd', 'adam'))
        learning_rate = st.selectbox(
            'learning_rate', ('constant', 'invscaling', 'adaptive'))

    with col2:
        activation = st.selectbox(
            'activation', ('identity', 'logistic', 'tanh', 'relu'))
        alpha = st.number_input('alpha', value=1e-4, format="%.4f")
        learning_rate_init = st.slider(
            'learning_rate_init', min_value=0.001, max_value=0.1, value=0.001)

    mlp_params = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'learning_rate_init': learning_rate_init
    }

elif classifier == "VAE":
    st.header("VAE")
    col1, col2 = st.columns(2)
    with col1:

        max_epoch = st.selectbox('max_epoch', options=[8, 16, 32, 64], index=0)
        batch_size = st.slider('max_epoch', min_value=20,
                               max_value=100, value=100)
        activation = st.selectbox(
            'activation', ('identity', 'logistic', 'tanh', 'relu'))
        alpha = st.number_input('alpha', value=1e-4, format="%.4f")
        learning_rate_init = st.slider(
            'learning_rate_init', min_value=1e-4, max_value=0.1, value=0.001, format="%.4f")
        solver = st.selectbox('solver', ('lbfgs', 'sgd', 'adam'))
        learning_rate = st.selectbox(
            'learning_rate', ('constant', 'invscaling', 'adaptive'))

    with col2:
        hidden_layers = st.slider(
            'hidden_layers', min_value=1, max_value=3, value=2)

        hidden_layer_sizes = []
        for i in range(hidden_layers):
            size = st.slider(
                f'Size of hidden layer {i+1}', min_value=1, max_value=100, value=50)
            hidden_layer_sizes.append(size)

    vae_params = {
        'hidden_layers': hidden_layers,
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'activation': activation,
        'solver': solver,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'learning_rate_init': learning_rate_init
    }


if classifier == 'KNN':
    selected_classifier_dict = knn_params
if classifier == 'SVC':
    selected_classifier_dict = svc_params
if classifier == 'Random Forest':
    selected_classifier_dict = rf_params
if classifier == 'MLP':
    selected_classifier_dict = mlp_params
if classifier == 'VAE':
    selected_classifier_dict = vae_params


with st.sidebar:
    on = st_toggle_switch(
        label="Hyperparameters Preview",
        key="switch_1",
        default_value=False,
        label_after=True,
        inactive_color="#D3D3D3",
        active_color="#11567f",
        track_color="#29B5E8",

    )
    if on:

        st.write(selected_classifier_dict)


# if col1.button("🙈 Preview"):
# st.write('Preview of the classifier:')

# st.write(selected_classifier)
# Add your training code/ function here!
add_vertical_space(3)
col1, col2, col3 = st.columns([2, 2, 1])

if col2.button("🙈 Training!"):

    switch_page("ActiveLearning")
