import streamlit as st
from PIL import Image
import altair as alt
import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
import pandas as pd
import base64
import glob
import streamlit.components.v1 as components
from PIL import Image
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import json
import sys
import classification.generic_classifer as generic
from classification.dataset import WbcDataset
from setup.config import login_statement, user_folder_config
import css_style
import time
import joblib
from post_score import roc_cells, f1_score_cells, get_accuracy_for_results, make_data_binary
import utils
from data_augmentation.data_augmentation import data_augmentation, CellsDataset

# user settings
USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
user_folder = USER_FOLDER_PATH

def read_model(USER_FOLDER_PATH, classifier_string):
    """
    This function loads a trained model from a joblib file and extracts its name.

    Parameters:
    - USER_FOLDER_PATH: a string representing the path to the user's directory.
    - classifier_string: a string representing the filename of the classifier to be loaded. 

    The function performs the following steps:
    1. Constructs the path to the model file by combining the user's directory, the model directory, 
       and the name of the model file. The model file's extension is changed from '.json' to '.joblib'.
    2. Loads the model from the joblib file using the constructed path.
    3. Extracts the name of the classifier from the filename.

    Returns:
    - clf: the loaded model.
    - classifier_name: a string representing the name of the classifier.

    Note: This function uses the `joblib.load` function to load the model from the joblib file. 
    The filename of the classifier is expected to start with the classifier's name, followed by 
    an underscore ('_').
    """
    model_dir = f'{USER_FOLDER_PATH}/model/'
    model_file = classifier_string.replace(".json", ".joblib")
    model_path = os.path.join(model_dir, model_file)
    clf = joblib.load(model_path)
    classifier_name = classifier_string.split("_")[0]
    return clf, classifier_name


def get_accuracy(USER_FOLDER_PATH, classifier, classifier_name):    
    """
    This function computes the accuracy of a given classifier on the test data.

    Parameters:
    - USER_FOLDER_PATH: a string representing the path to the user's directory.
    - classifier: the trained model to be used for prediction.
    - classifier_name: a string representing the name of the classifier. 

    The function performs the following steps:
    1. Loads the test data from the specified path. If the path does not exist, the function 
       raises a warning and terminates the program. 
    2. Depending on the classifier's name, it creates a dataset object with appropriate parameters.
    3. Extracts the features (X_test) and labels (y_test) from the test dataset.
    4. Computes the accuracy of the classifier on the test data by comparing the classifier's 
       predictions to the true labels. The accuracy is rounded to two decimal places and converted 
       into a percentage.

    Returns:
    - accuracy: a float representing the classifier's accuracy on the test data.

    Note: This function uses the classifier's `score` method to compute the accuracy.
    """
    # Calling the already trained model
    clf = classifier
    # Split dataset and train model(replace by reading the model file)
    test_data_path = f'{USER_FOLDER_PATH}/dataset/test'
    if not os.path.exists(test_data_path):
        st.warning("There is no labeled data, please back to the labeling page.")
        sys.exit('Program terminated.')
    if classifier_name == "VAE":
        dataset_test = WbcDataset(dir=test_data_path, split='all',
                                    transform=None, download=False, need_label=True, resize=True, need_feature=False)
    elif classifier_name == "CNN":
        dataset_test = WbcDataset(dir=test_data_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=False)
    else:
        dataset_test = WbcDataset(dir=test_data_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=True)
    X_test, y_test = dataset_test.create_X_y()
    
    # Show the reults
    accuracy = round(clf.score(X_test, y_test),2)*100
    
    return accuracy


def get_uncertainty_and_filename(USER_FOLDER_PATH, classifier, classifier_name):
    """
    This function computes the uncertainty of a given classifier on unlabeled data and 
    returns the data, uncertainty scores, ranked scores, and the file names.

    Parameters:
    - USER_FOLDER_PATH: a string representing the path to the user's directory.
    - classifier: the trained model to be used for prediction.
    - classifier_name: a string representing the name of the classifier. 

    The function performs the following steps:
    1. Loads the unlabeled data from the specified path. 
    2. Depending on the classifier's name, it creates a dataset object with appropriate parameters.
    3. Extracts the features (X_unlabeled) from the unlabeled dataset.
    4. Computes the uncertainty of the classifier on the unlabeled data using the classifier's 
       `predict_uncertainty` method.
    5. Ranks the uncertainty scores in descending order and assigns a rank to each score.
    6. Retrieves the file names of the images in the unlabeled dataset.

    Returns:
    - data_unlabeled: a dataset object containing the unlabeled data.
    - uncertainty_sorting: a list of uncertainty scores corresponding to the unlabeled data.
    - ranked_scores: a list of ranks corresponding to the uncertainty scores.
    - list_name: a list of file names corresponding to the unlabeled data.

    Note: The `predict_uncertainty` method is assumed to be a part of the classifier's 
    implementation. It should return a measure of the classifier's uncertainty or 
    confidence in its predictions.
    """
    clf = classifier
    unlabeled_path = f'{USER_FOLDER_PATH}/dataset/train_unlabeled'
    if classifier_name == "VAE":
        data_unlabeled = WbcDataset(dir=unlabeled_path, split='all',
                                    transform=None, download=False, need_label=False, resize=True, need_feature=False)
    elif classifier_name == "CNN":
        data_unlabeled = WbcDataset(dir=unlabeled_path, split='all',
                                    transform=None, download=False, need_label=False, resize=False, need_feature=False)
    else:
        data_unlabeled = WbcDataset(dir=unlabeled_path, split='all',
                                    transform=None, download=False, need_label=False, resize=False, need_feature=True)

    X_unlabeled, _ = data_unlabeled.create_X_y()
    uncertainty_sorting = clf.predict_uncertainty(X_unlabeled)
    
    # calculate uncertainty ranking 
    sorted_indexes = sorted(range(len(uncertainty_sorting)), key=lambda k: uncertainty_sorting[k], reverse=True)
    ranked_scores = [sorted_indexes.index(i) + 1 for i in range(len(uncertainty_sorting))]

    # get filename list
    list_name = []
    for idx in range(len(data_unlabeled)):
        _, _, file_name = data_unlabeled[idx]
        list_name.append(file_name)
    print(list_name)
    return data_unlabeled, uncertainty_sorting, ranked_scores, list_name


def get_probability(data_unlabeled, classifier):
    X_unlabeled, _ = data_unlabeled.create_X_y()
    clf = classifier
    probabilities = clf.predict_proba(X_unlabeled)
    return probabilities

def load_test_dataset(USER_FOLDER_PATH, classifier_name):
    test_data_path = f'{USER_FOLDER_PATH}/dataset/test'
    # test_dataset generate
    if not os.path.exists(test_data_path):
        st.warning("There is no labeled data, please back to the labeling page.")
        sys.exit('Program terminated.')
    if classifier_name == "VAE":
        dataset_test = WbcDataset(dir=test_data_path, split='all',
                                    transform=None, download=False, need_label=True, resize=True, need_feature=False)
    elif classifier_name == "CNN":
        dataset_test = WbcDataset(dir=test_data_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=False)
    else:
        dataset_test = WbcDataset(dir=test_data_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=True)
    
    labels = []
    for i in range(len(dataset_test)):
        _, label, _ = dataset_test[i]
        labels.append(label)
        
    X_test, y_test = dataset_test.create_X_y()
    y_test_bin = make_data_binary(y_test, labels)
    return X_test, y_test, y_test_bin, labels
    
    
def results_output(USER_FOLDER_PATH, classifier, classifier_string, X_test, y_test, y_test_bin, labels, AL_time):
    """
    The function results_output is used to generate and save the results of a classifier's performance. It accepts the following parameters:
    Parameters:
    - USER_FOLDER_PATH: The path to the folder where the user's files are stored. The function will create a 'results' subdirectory within this folder if it does not already exist.
    - classifier: The trained classifier model which will be used to make predictions.
    - classifier_string: A string representing the name of the classifier. It will be used to generate the filename for the results.
    - X_test: The test data that the classifier will make predictions on.
    - y_test: The actual labels of the test data.
    - y_test_bin: The binary format of the test labels, used for computing ROC curves.
    - AL_time: An integer representing the active learning step, used in generating the title for the ROC curve.
    """
    
    clf = classifier
    # makedir if not exists
    if not os.path.exists(f'{USER_FOLDER_PATH}/results/'):
        os.makedirs(f'{USER_FOLDER_PATH}/results/')
    # load existing data if file exists
    post_scores_path = f'{USER_FOLDER_PATH}/results/{classifier_string.replace(".json", "")}_postprocessing_scores.json'
    if os.path.exists(post_scores_path):
        existing_data = utils.load_json(post_scores_path)
        # read the values
        fpr_list = existing_data['fpr']
        tpr_list = existing_data['tpr']
        roc_auc_list = existing_data['roc_auc']
        f1_score_class_list = existing_data['f1_score_class']
        f1_score_overall_list = existing_data['f1_score_overall']
        acc_list = existing_data['acc']
        int2cells_dict = existing_data['int2cells_dict']
    else:
        # initialize lists
        f1_score_class_list = []
        f1_score_overall_list = []
        acc_list = []
        fpr_list = []
        tpr_list = []
        roc_auc_list = []

    cells2int_dict = {'wbc':0, 'rbc':1, 'plt':2, 'agg':3, 'oof':4}
    int2cells_dict = {v: k for k, v in cells2int_dict.items()}
    
    y_scores_test = clf.predict_proba(X_test)
    
    # ROC curve
    title = f'ROC curve with {AL_time+1} activate learning step'
    ROC_pic_path = f'ROC_curve_{AL_time+1}_activate_learning_step.png'
    fpr, tpr, roc_auc = roc_cells(y_test_bin, y_scores_test, title, save_pic=False, save_path=ROC_pic_path, int2cells_dict=int2cells_dict)
    
    ## values in dict cannot be saved in json file, convert to list
    for key in fpr:
        fpr[key] = fpr[key].tolist()

    for key in tpr:
        tpr[key] = tpr[key].tolist()

    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)

    # f1 score
    f1_score_class, f1_score_overall = f1_score_cells(y_test, clf.predict(X_test), labels)
    # f1_score_class_str = [f'{int2cells_dict[i]}: {f1_score_class[i]:.3f}' for i in range(5)]
    
    f1_score_class_list.append(f1_score_class)
    f1_score_overall_list.append(f1_score_overall)
    
    # acc
    acc = get_accuracy_for_results(y_test, clf.predict(X_test))
    acc_list.append(acc)
    
    # save as json.file
    json_dicts = {'fpr':fpr_list, 'tpr':tpr_list, 'roc_auc':roc_auc_list, 
                    'f1_score_class':f1_score_class_list, 
                    'f1_score_overall':f1_score_overall_list, 'acc':acc_list, 
                    'int2cells_dict': int2cells_dict}
    utils.save_json(json_dicts, post_scores_path)

def images_moving_and_augmentation(USER_FOLDER_PATH, unlabeled_dataset, image_name_list,label_list,labeled_folder):
    """
    Move selected images from an unlabeled dataset to a labeled dataset and perform data augmentation.

    Parameters:
    - USER_FOLDER_PATH (str): Base directory path for the user's data.
    - unlabeled_dataset (Dataset): The dataset containing unlabeled cell images.
    - image_name_list (list of str): List of image names to be moved and augmented.
    - label_list (list): Corresponding labels for images in 'image_name_list'.
    - labeled_folder (str): Directory where augmented images will be saved.

    Workflow:
    1. Iterates over each image name in 'image_name_list'.
    2. Moves the image from the unlabeled dataset to an 'augment_folder' directory.
    3. Calls the 'data_augmentation' function to augment the moved images.
    4. The augmented images are saved in 'labeled_folder'.
    5. Finally, clears the 'augment_folder' directory for future use.

    Returns:
    None. The function performs operations in-place and saves the augmented images to the specified directory.

    Notes:
    - This function assumes the existence of a 'data_augmentation' function and a 'CellsDataset' class.
    - This function ensures that the 'augment_folder' directory is clean after its operations, so it's ready for subsequent uses.
    """

    augment_folder = f'{USER_FOLDER_PATH}/dataset/train_augment'
    if not os.path.exists(augment_folder):
        os.makedirs(augment_folder)

    N = len(image_name_list)
    for i in range(N):
        name = image_name_list[i]
        label = label_list[i]
        
        unlabeled_dataset.rename_and_move_image(name,label,augment_folder)
        
    data_augmentation(root_dir=augment_folder, png_save_root_dir=labeled_folder, save_dataset=False, dataset_save_path=None,CellsDataset=CellsDataset, 
                num_augmentation=4, suffix='png')
    
    # delete files in augment_dataset for next time AL
    for filename in os.listdir(augment_folder):
        file_path = os.path.join(augment_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}') 
    
    
def retrain_get_accuracy(USER_FOLDER_PATH, classifier_name, classifier_string, classifier, AL_time):
    """
    This function retrains a given classifier on the labeled training data and computes its accuracy on the test data.

    Parameters:
    - USER_FOLDER_PATH: a string representing the path to the user's directory.
    - classifier_name: a string representing the name of the classifier.
    - classifier: the model to be retrained.

    The function performs the following steps:
    1. Loads the labeled training and test data from the specified paths.
    2. Depending on the classifier's name, it creates dataset objects for the training and test data with appropriate parameters.
    3. Extracts the features (X_train, X_test) and labels (y_train, y_test) from the training and test datasets.
    4. Retrains the classifier on the training data.
    5. Computes the classifier's accuracy on the test data.

    Returns:
    - clf: the retrained classifier.
    - accuracy: the classifier's accuracy on the test data (as a percentage).

    Note: The function assumes that the classifier has 'fit' and 'score' methods as part of its implementation. 
    The 'fit' method is used for retraining the classifier on the training data, 
    and the 'score' method is used for computing the classifier's accuracy on the test data.
    """
    # for the result
    if AL_time == 0:
        X_test, y_test, y_test_bin, labels = load_test_dataset(USER_FOLDER_PATH, classifier_name)
        results_output(USER_FOLDER_PATH, classifier, classifier_string, X_test, y_test, y_test_bin, labels, AL_time)
        
    # dataset
    train_path = f'{USER_FOLDER_PATH}/dataset/train_labeled'
    if not os.path.exists(train_path):
        st.warning("There is no labeled data, please back to the labeling page.")
        sys.exit('Program terminated.')
    test_path = f'{USER_FOLDER_PATH}/dataset/test'
    if not os.path.exists(test_path):
        st.warning("There is no labeled data, please back to the labeling page.")
        sys.exit('Program terminated.')
    if classifier_name == "VAE":
        dataset_train = WbcDataset(dir=train_path, split='all',
                                    transform=None, download=False, need_label=True, resize=True, need_feature=False)
        dataset_test = WbcDataset(dir=test_path, split='all',
                                    transform=None, download=False, need_label=True, resize=True, need_feature=False)
        
    elif classifier_name == "CNN":
        dataset_train = WbcDataset(dir=train_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=False)
        dataset_test = WbcDataset(dir=test_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=False)
        
    else:
        dataset_train = WbcDataset(dir=train_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=True)
        dataset_test = WbcDataset(dir=test_path, split='all',
                                    transform=None, download=False, need_label=True, resize=False, need_feature=True)
    X_train, y_train = dataset_train.create_X_y()
    X_test, y_test = dataset_test.create_X_y()
    
    # retrain model
    clf = classifier
    clf.fit(X_train, y_train)
    accuracy = round(clf.score(X_test, y_test),2)*100
    return clf, accuracy