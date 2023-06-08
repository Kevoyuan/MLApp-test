import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Title
st.title('Classifier Parameter Selection')

# Dropdown for classifier selection
classifier = st.selectbox(
    'Select classifier',
    ('KNN', 'SVC', 'Random Forest', 'MLP')
)

# Depending on the classifier selected, display the appropriate options
if classifier == 'KNN':
    n_neighbors = st.slider('n_neighbors', min_value=2, max_value=10, value=5)
    weights = st.selectbox('weights', ('uniform', 'distance'))
    algorithm = st.selectbox('algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
    metric = st.selectbox('metric', ('l1', 'l2'))
    
    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        metric=metric
    )

elif classifier == 'SVC':
    C = st.slider('C', min_value=0.0, max_value=1.0, value=1.0)
    kernel = st.selectbox('kernel', ('linear', 'rbf', 'sigmoid', 'precomputed'))
    gamma = st.selectbox('gamma', ('scale', 'auto'))

    classifier = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma
    )

elif classifier == 'Random Forest':
    n_estimators = st.slider('n_estimators', min_value=1, max_value=200, value=100)
    criterion = st.selectbox('criterion', ('gini', 'entropy', 'log_loss'))
    max_depth = st.slider('max_depth', min_value=1, max_value=100, value=10)
    max_features = st.selectbox('max_features', ('sqrt', 'log2', None))

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features
    )

elif classifier == 'MLP':
    hidden_layer_sizes = st.slider('hidden_layer_sizes', min_value=1, max_value=100, value=50)
    activation = st.selectbox('activation', ('identity', 'logistic', 'tanh', 'relu'))
    solver = st.selectbox('solver', ('lbfgs', 'sgd', 'adam'))
    alpha = st.slider('alpha', min_value=0.0001, max_value=0.01, value=0.0001)
    learning_rate = st.selectbox('learning_rate', ('constant', 'invscaling', 'adaptive'))
    learning_rate_init = st.slider('learning_rate_init', min_value=0.001, max_value=0.1, value=0.001)

    classifier = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_sizes,),
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init
    )

