import h5py
from matplotlib import pyplot as plt
import matplotlib.image
from PIL import Image
from tqdm import tqdm
import numpy as np
import re

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import sklearn
import sys, os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join(current_path, '..') ) 
import utils

def get_accuracy_for_results(y_true, y_pred):

    '''
    Compute accuracy score
        Parameters:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            accuracy: accuracy score
    '''
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def check_dim(data):
    '''
    Check the dimension of data if
        Parameters:
            data: data to be checked
        Returns:
            None
    '''
    
    if data.ndim == 1:
        N = data.shape[0]
        return data.reshape(1, N)
    elif data.ndim > 2:
        raise ValueError('Data dimension should be 1 or 2')
    else:
        return data
    
    # if len(data.shape) == 2:
    #     data = data.reshape(96, 96, -1)
    # elif len(data.shape) == 3:


def make_data_binary(y_test, labels):
    '''
    Convert labels to binary labels
        Parameters:
            y_test: labels of test data
            labels: all labels
        Returns:
            y_test_bin: onehot encoding of test data
    '''
    
    y_test_bin = label_binarize(y_test, classes=np.unique(labels))
    return y_test_bin


def roc_cells(y_test_bin, y_scores, title='ROC Curve', save_pic=False, show_pic=False, save_path='test.png', int2cells_dict=None):
    '''
    Compute ROC curve and ROC area for each class and overall
        Parameters:
            y_test_bin: binary labels of test data
            y_scores: predicted probabilities of test data
        Returns:
            fpr: false positive rate
            tpr: true positive rate
            roc_auc: area under curve
    '''
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test_bin = np.array(y_test_bin)
    y_scores = np.array(y_scores)
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm']

    for i, color in zip(range(5), colors):
        
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (AUC = {1:0.2f})'.format(int2cells_dict[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_pic:
        plt.savefig(save_path)
    if show_pic:
        plt.show()
    return fpr, tpr, roc_auc



def f1_score_cells(y_true, y_pred, labels):
    '''
    Compute F1 score for each class 
        Parameters:
            y_test: labels of test data
            y_pred: predicted labels of test data
        Returns:
            None
    '''
    y_test_bin = make_data_binary(y_true, labels)
    y_true_bin = make_data_binary(y_pred, labels)
    f1_class = f1_score(y_test_bin, y_true_bin, average=None).tolist()
    f1_overall = f1_score(y_true, y_pred, average='weighted')
    return f1_class, f1_overall


if __name__ == '__main__':
    
    import time
    
    from sklearn.linear_model import LogisticRegression
    
    cells2int_dict = {'wbc':0, 'rbc':1, 'plt':2, 'agg':3, 'oof':4}
    int2cells_dict = {v: k for k, v in cells2int_dict.items()}
    
    data_root = r'W:\samples\prediction'
    
    # start_time = time.time()
    # images, labels = utils.read_dataset_from_png(data_root, suffix='.png', label_to_int=True, cells2int_dict=cells2int_dict)
    # print(f'time to load png: {time.time() - start_time}')
    # images = images.reshape(96*96, -1).T
    
    # np.savez('data.npz', images=images, labels=labels)
    start_time = time.time()
    data = np.load(r'W:\data_augmentation\data.npz')

    images = data['images']
    labels = data['labels']
    print(f'time to load png: {time.time() - start_time}')
    
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.9, random_state=42)
    
    X_test, X_train_ac, y_test, y_train_ac = train_test_split(X_test, y_test, test_size=0.2, random_state=35)
    
    ## split active learning data into 3 folds
    X_train_ac = np.array_split(X_train_ac, 3)
    y_train_ac = np.array_split(y_train_ac, 3)
    
    y_test_bin = make_data_binary(y_test, labels)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    # y_scores = classifier.predict_proba(X_test)

    num_activate_learning = 3
    classifier_ac = sklearn.base.clone(classifier)
    
    f1_score_class_list = []
    f1_score_overall_list = []
    
    acc_list = []
    
    fpr_list = []
    tpr_list = []
    roc_auc_list = []
    
    ## incremental dataset for active learning
    X_train_ac_total = X_train
    y_train_ac_total = y_train  
    
    # np.savetxt("postprocessing_X_train.csv", X_train_ac_total, delimiter=",")
    # np.savetxt("postprocessing_y_train.csv", X_train_ac_total, delimiter=",")
    
    # np.savetxt("postprocessing_X_test.csv", X_test, delimiter=",")
    # np.savetxt("postprocessing_y_test.csv", y_test, delimiter=",")
    
    # y_test_bin = make_data_binary(y_test, labels)
    # np.savetxt(f"postprocessing_y_test_bin.csv", y_test_bin, delimiter=",")
    
    save_pic =  False
    for i in range(num_activate_learning): 
        

        X_train_ac_temp = np.vstack(X_train_ac[i])
        y_train_ac_temp = np.hstack(y_train_ac[i])
        
        X_train_ac_total = np.vstack((X_train_ac_total, X_train_ac_temp))
        y_train_ac_total = np.hstack((y_train_ac_total, y_train_ac_temp))
        
        
        # np.savetxt(f"postprocessing_X_train_AL_step_{i+1}.csv", X_train_ac_total, delimiter=",")
        # np.savetxt(f"postprocessing_y_train_AL_step_{i+1}.csv", y_train_ac_total, delimiter=",")
    
        # X_train_ac_temp = check_dim(X_train_ac_temp)
        # y_train_ac_temp = check_dim(y_train_ac_temp)
        
        classifier_ac.fit(X_train_ac_total, y_train_ac_total)
        
        # test on oracle test data
        y_scores_test = classifier_ac.predict_proba(X_test)
        

        
        title = f'ROC curve with {i+1} activate learning step'
        ROC_pic_path = f'ROC_curve_{i+1}_activate_learning_step.png'
        fpr, tpr, roc_auc = roc_cells(y_test_bin, y_scores_test, title, save_pic=save_pic, save_path=ROC_pic_path)
        
        ## values in dict cannot be saved in json file, convert to list
        for key in fpr:
            fpr[key] = fpr[key].tolist()
    
        for key in tpr:
            tpr[key] = tpr[key].tolist()
    
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        
        f1_score_class, f1_score_overall = f1_score_cells(y_test, classifier_ac.predict(X_test), labels)
        f1_score_class_str = [f'{int2cells_dict[i]}: {f1_score_class[i]:.3f}' for i in range(5)]
        
        f1_score_class_list.append(f1_score_class)
        f1_score_overall_list.append(f1_score_overall)
        
        print(f'with {i+1} activate learning step \n')
        print(f'f1 score: f1_score_class: {f1_score_class_str}')
        print(f'overfall f1 score: {f1_score_overall}')
        
        acc = accuracy_score(y_test, classifier_ac.predict(X_test))
        acc_list.append(acc)
        
        print(f'test accuracy: {acc}\n\n')
        
    json_dicts = {'fpr':fpr_list, 'tpr':tpr_list, 'roc_auc':roc_auc_list, 
                    'f1_score_class':f1_score_class_list, 
                    'f1_score_overall':f1_score_overall_list, 'acc':acc_list, 
                    'int2cells_dict': int2cells_dict}
    post_scores_path = f'D:\GitHub\Group05\postprocessing/postprocessing_scoresV02.json'
    utils.save_json(json_dicts, post_scores_path)