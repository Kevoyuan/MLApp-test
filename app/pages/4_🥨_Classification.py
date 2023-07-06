import streamlit as st
from page_config import login_statement
from streamlit_option_menu import option_menu
from streamlit_toggle import st_toggle_switch
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
from css_style import generate_menu_styles
# import classification.generic_classifer as generic
# from classification.dataset import WbcDataset
import json
from config import USER_FOLDER_PATH, SAVE_ROOT_PATH

with st.sidebar:
    login_statement()


classifier = option_menu('', ['KNN', 'SVC', 'Random Forest', 'MLP', 'VAE'],
                         icons=['-', '-', '-', '-', '-'],
                         menu_icon="-", default_index=0, orientation="horizontal",
                         styles=generate_menu_styles())


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
        metric = st.selectbox('metric', ('l1', 'l2'), index=1)

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
            'kernel', ('rbf', 'linear', 'sigmoid', 'precomputed'), index=0)

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
                              max_value=100, value=None)
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
        solver = st.selectbox('solver', ('lbfgs', 'sgd', 'adam'), index=2)
        learning_rate = st.selectbox(
            'learning_rate', ('constant', 'invscaling', 'adaptive'))

    with col2:
        activation = st.selectbox(
            'activation', ('identity', 'logistic', 'tanh', 'relu'), index=3)
        alpha = st.number_input('alpha', value=1e-4, format="%.4f")
        learning_rate_init = st.select_slider('learning_rate_init', options=[
                                              '0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05', '0.1'])

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

        batch_size = st.select_slider('batch_size', options=[8, 16, 32, 64])
        max_epoch = st.slider('max_epoch', min_value=20,
                               max_value=100, value=100)
        activation = st.selectbox(
            'activation', ('identity', 'logistic', 'tanh', 'relu'), index=3)
        # alpha = st.number_input('alpha', value=1e-4, format="%.4f")
        learning_rate_init = st.select_slider('learning_rate_init', options=[
                                              0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

        solver = st.selectbox('solver', ('lbfgs', 'sgd', 'adam'), index=2)
        learning_rate = st.selectbox(
            'learning_rate', ('constant', 'invscaling', 'adaptive'), index=0)

    with col2:
        hidden_layers = st.slider(
            'hidden_layer_num', min_value=1, max_value=3, value=2)

        hidden_layer_sizes = []
        for i in range(hidden_layers):
            size = st.select_slider(
                f'Size of hidden layer {i+1}', options=[8, 16, 32, 64, 128], value=64)
            hidden_layer_sizes.append(size)

    vae_params = {
        'hidden_layer_num': hidden_layers,
        "hidden_layer_sizes": hidden_layer_sizes,
        'max_epoch': max_epoch,
        'batch_size': batch_size,
        'activation': activation,
        'solver': solver,
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

add_vertical_space(3)
col1, col2, col3 = st.columns([2, 2, 1])

if col2.button("🙈 Pretrain!"):
    ## The pretrain function here! 
    clf = generic.Classifier_cells(classifier, **selected_classifier_dict)
    # Split dataset
    labeled_path = f'{USER_FOLDER_PATH}/dataset/labeled'
    if classifier == "VAE":
        dataset_train = WbcDataset(dir=labeled_path, split='train', transform=None, download=False, need_label=True, resize=True)
        dataset_test = WbcDataset(dir=labeled_path, split='test', transform=None, download=False, need_label=True, resize=True)
    else:
        dataset_train = WbcDataset(dir=labeled_path, split='train', transform=None, download=False, need_label=True, resize=False)
        dataset_test = WbcDataset(dir=labeled_path, split='test', transform=None, download=False, need_label=True, resize=False)

    # Training
    X, y = dataset_train.create_X_y()
    clf.fit(X, y)
    # Testing
    X, y = dataset_test.create_X_y()
    # Accuracy
    print(clf.score(X, y))
    switch_page("ActiveLearning")
