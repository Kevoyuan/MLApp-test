import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from streamlit_toggle import st_toggle_switch
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)
classifier_titles = ['KNN', 'SVC', 'Random Forest', 'MLP']

classifier = st.tabs(classifier_titles)


with classifier[0]:

    # Depending on the classifier selected, display the appropriate options
    st.header("KNN")
    # n_neighbors = st.slider('n_neighbors', min_value=2, max_value=10, value=5)
    # weights = st.selectbox('weights', ('uniform', 'distance'))
    # algorithm = st.selectbox('algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    # metric = st.selectbox('metric', ('l1', 'l2'))
    col1, col2 = st.columns(2)

    with col1:
        n_neighbors = st.slider(
            'n_neighbors', min_value=2, max_value=10, value=5)

    with col2:
        weights = st.selectbox('weights', ('uniform', 'distance'))
    col3, col4 = st.columns(2)
    with col3:
        algorithm = st.selectbox(
            'algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))

    with col4:
        metric = st.selectbox('metric', ('l1', 'l2'))

    knn_clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric=metric
    )


with classifier[1]:
    st.header("SVC")
    col1, col2 = st.columns(2)
    with col1:
        C = st.slider('C', min_value=0.0, max_value=1.0, value=1.0)
    with col2:
        kernel = st.selectbox(
            'kernel', ('linear', 'rbf', 'sigmoid', 'precomputed'))


    col1, col2 = st.columns(2)
    with col1:
        gamma = st.selectbox('gamma', ('scale', 'auto'))

    svc_clf = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma
    )

with classifier[2]:
    st.header("Random Forest")
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider(
            'n_estimators', min_value=1, max_value=200, value=100)
    with col2:
        criterion = st.selectbox('criterion', ('gini', 'entropy', 'log_loss'))

    col1, col2 = st.columns(2)
    with col1:
        max_depth = st.slider('max_depth', min_value=1,
                              max_value=100, value=10)
    with col2:
        max_features = st.selectbox('max_features', ('sqrt', 'log2', None))

    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features
    )

with classifier[3]:
    st.header("MLP")
    col1, col2 = st.columns(2)

    with col1:
        hidden_layer_sizes = st.slider(
            'hidden_layer_sizes', min_value=1, max_value=100, value=50)
    with col2:
        activation = st.selectbox(
            'activation', ('identity', 'logistic', 'tanh', 'relu'))
    col1, col2 = st.columns(2)

    with col1:
        solver = st.selectbox('solver', ('lbfgs', 'sgd', 'adam'))
    with col2:
        alpha = st.number_input('alpha', value=1e-4, format="%.4f")
    col1, col2 = st.columns(2)

    with col1:
        learning_rate = st.selectbox(
            'learning_rate', ('constant', 'invscaling', 'adaptive'))
    with col2:
        learning_rate_init = st.slider(
            'learning_rate_init', min_value=0.001, max_value=0.1, value=0.001)

    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_sizes,),
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init
    )

add_vertical_space(3)

# Create three columns
col1, col2, col3 = st.columns([2, 2, 1])
if col2.button("🙈 Training!"):
    switch_page("Training")
