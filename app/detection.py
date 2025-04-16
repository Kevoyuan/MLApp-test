import copy
import streamlit as st
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import pickle
import cv2
import os
import time
import supervision as sv
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from css_style import generate_segmentation_progress_bar
from classification.feature_extraction import calculate_contour_properties
import random
import string
import pandas as pd
import json
import joblib
from PIL import Image, ImageDraw
ImageDraw.LOAD_TRUNCATED_IMAGES = True
# from user_folder_util import handle_user_folder, create_directory_if_not_exists
from setup.config import login_statement, user_folder_config

# for BBoxWidget
import base64
# from jupyter_bbox_widget import BBoxWidget

st.cache_resource.clear()
USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
user_folder = USER_FOLDER_PATH
save_root = SAVE_ROOT_PATH
print("user_folder in detection: ", user_folder)

# Description:
# you need to locate to your main project folder,
# where you need a folder named: sam_weights, for storing the pretrained weights.
# The pretrained weights can be found in the group_share folder

# Save directory example
# - data
# - - example_dir
# - - - original.png
# - - - segmented.png
# - - - segmentation.pkl
# - - - boxes
# - - - - box01.png
# - - - - box02.png
# - - - - ...
# Boxes of new uploaded imagtes are additionally saved to the unlabeled_xxx folder with a randomized name

# os.chdir('../')
# HOME = os.getcwd()
# SAVE_ROOT_PATH = "Z:/APP_test"

# The pretrained weights are stored in the sam_weights/ folder
parameters_segmentation_stack = {
    'CHECKPOINT_PATH': os.path.join("./", "sam_weights", "sam_vit_b_01ec64.pth"),
    'MODEL_TYPE': "vit_b",
    'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}

url = 'https://dl.fbaipublicfiles.com/segment_anything/'
ckpt_dir = './sam_weights/'
model_type = parameters_segmentation_stack['MODEL_TYPE']
ckpt_file = f"sam_{model_type}_01ec64.pth"

# If the checkpoint file does not exist, download it
if not os.path.exists(os.path.join(ckpt_dir, ckpt_file)):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    st.sidebar.success(f"Downloading {model_type} model...")
    urllib.request.urlretrieve(url + ckpt_file, os.path.join(ckpt_dir, ckpt_file))
    st.sidebar.success("Download complete.")
    st.experimental_rerun()

def get_image_masks_with_label(user_folder,image_name, model_name):
    name = os.path.splitext(os.path.basename(image_name))[0]
    save_root = os.path.join(user_folder, 'prediction', name)
    os.makedirs(save_root, exist_ok=True)
    image_path = os.path.join(save_root, image_name)
    masks, annotated_image, image_bbox = get_image_masks(save_root, 0, model_name, image_path, save_as_pkl=False, save_annotated=False,\
                                                          return_elapsed_time=False, return_annotated=True, return_bbox=True)
    bbox_name = os.path.splitext(os.path.basename(image_name))[0] + '_bbox.png'
    bbox_path = os.path.join(save_root, bbox_name)
    cv2.imwrite(bbox_path, image_bbox)
    model = joblib.load(os.path.join(user_folder, 'model', model_name))
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    data = []
    i = 0
    for mask in masks:
        mask_save = mask.copy()
        mask = np.array(mask).astype(np.uint8)
        box = mask2box(image_rgb, mask, filter_white=False)
        box_bgr = cv2.cvtColor(box, cv2.COLOR_RGB2BGR)
        box_path = os.path.join(save_root, 'boxes')
        if not os.path.exists(box_path):
            os.makedirs(box_path)
        if box is not None:
            if model_name.startswith('VAE'):
                cv2.imwrite(os.path.join(save_root, 'boxes', f'box_{i}.png'), box_bgr)   
                box = cv2.resize(box, (64, 64))
                box = np.transpose(box, (2, 0, 1)).reshape(1, 3, 64, 64) 
            elif model_name.startswith('CNN'):
                box_path = os.path.join(save_root, 'boxes')
                cv2.imwrite(os.path.join(box_path, f'box_{i}.png'), box_bgr)
                box = cv2.resize(box, (96, 96))
                box = np.transpose(box, (2, 0, 1)).reshape(1, 3, 96, 96)
            else:
                cv2.imwrite(os.path.join(save_root, 'boxes', f'box_{i}.png'), box_bgr)   
                box_path = os.path.join(save_root, 'boxes', f'box_{i}.png')
                box = calculate_contour_properties(box_path, return_list=True).reshape(1, -1)
            pred = model.predict(box)[0]
            i += 1
            # Put the label and mask to dataframe
            if pred == 0:
                label = 'WBC'
            elif pred == 1:
                label = 'RBC'
            elif pred == 2:
                label = 'PLT'
            elif pred == 3:
                label = 'AGG'
            elif pred == 4:
                label = 'OOF'
            else:
                raise ValueError("Invalid label")
            item = {'masks': mask_save, 'label': label}
            data.append(item)
    df = pd.DataFrame(data)
    csv_name = os.path.splitext(os.path.basename(image_name))[0] + '.csv'
    csv_path = os.path.join(save_root, csv_name)
    df.to_csv(csv_path, index=False)
    mask_path = os.path.join(save_root, os.path.splitext(os.path.basename(image_name))[0] + '.pkl')
    with open(mask_path, 'wb') as save_segment:
        pickle.dump(masks, save_segment)
    return mask_path, csv_path, bbox_path

def get_image_masks(save_root, i, dir, image_name, save_as_pkl=False, save_annotated=False, return_elapsed_time=False,
                    return_annotated=False, para=parameters_segmentation_stack, return_bbox=False):
    """
    returns masks of given image
    :param dir: name of the save directory
    :param save_as_pkl: whether to save pickled file
    :param save_annotated: whether to save annotated image
    :param return_elapsed_time: whether to return elapsed time
    :param return_annotated: whether to return annotated image
    :param para: parameters for sam model
    :return: masks, original or as pickled file
    """
    start_time = time.time()
    gradient = "linear-gradient(to right, #4cd964, #5ac8fa, #007aff, #34aadc, #5856d6, #ff2d55)"
    total_steps = 14
    progress_placeholder = st.empty()

    sam = sam_model_registry[para['MODEL_TYPE']](
        checkpoint=para['CHECKPOINT_PATH']).to(device=para['DEVICE'])
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        1, total_steps, gradient, i), unsafe_allow_html=True)

    # Fine-tuning SAM to yield better segmentation
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        # points_per_side=32,
        # pred_iou_thresh=0.86,
        # stability_score_thresh=0.92,
        # crop_n_layers=1,
        # crop_n_points_downscale_factor=2,
        # # min_mask_region_area=100,
    )
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        2, total_steps, gradient, i), unsafe_allow_html=True)

    # assert os.path.exists(os.path.join(SAVE_ROOT_PATH, dir)
    #                       ), "Directory does not exist"
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        3, total_steps, gradient, i), unsafe_allow_html=True)

    image_bgr = cv2.imread(image_name)
    time.sleep(0.1)
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        10, total_steps, gradient, i), unsafe_allow_html=True)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        13, total_steps, gradient, i), unsafe_allow_html=True)

    sam_result = mask_generator.generate(image_rgb)
    assert len(sam_result) > 0, "Sam model ran unsuccessfully."

    # Identify and delete the sam_results that are of low quality
    sam_result = filter_out_overlapping_segmentation(sam_result)
    sam_result, removed = filter_and_extract_boxes(save_root, image_bgr, sam_result, dir=dir, save_boxes=False)

    progress_placeholder.markdown(generate_segmentation_progress_bar(
        13, total_steps, gradient, i), unsafe_allow_html=True)

    if len(sam_result) >= 1:
        masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)]
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(sam_result=sam_result)
        print(sam_result[0])
        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        box_annotator = sv.BoxAnnotator()
        image_bbox = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
    else:
        masks = []
        annotated_image = image_bgr.copy()
        image_bbox = image_bgr.copy()

    # if len(sam_result) > 1:
    #
    #     sam_result, removed = filter_and_extract_boxes(
    #         image_bgr, sam_result, dir=dir, save_boxes=False)
    #     # If there are still boxes after filtering
    #     if len(sam_result) > 0:
    #         masks = [mask['segmentation'] for mask in sorted(
    #             sam_result, key=lambda x: x['area'], reverse=True)]
    #         mask_annotator = sv.MaskAnnotator()
    #         detections = sv.Detections.from_sam(sam_result=sam_result)
    #         annotated_image = mask_annotator.annotate(
    #             scene=image_bgr.copy(), detections=detections)
    #         box_annotator = sv.BoxAnnotator()
    #         image_bbox = box_annotator.annotate(
    #             scene=image_bgr.copy(), detections=detections, skip_label=True)
    #     else:
    #         masks = []
    #         annotated_image = image_bgr.copy()
    #         image_bbox = image_bgr.copy()
    # else:
    #     masks = []
    #     annotated_image = image_bgr.copy()
    #     image_bbox = image_bgr.copy()

    if save_annotated:
        cv2.imwrite(os.path.join(save_root, dir,
                                 'segmented.png'), annotated_image)
        cv2.imwrite(os.path.join(save_root, dir, 'bbox.png'), image_bbox)
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        13, total_steps, gradient, i), unsafe_allow_html=True)

    # generate_head2head_comparison(dir, image_name, annotated_image)
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        13, total_steps, gradient, i), unsafe_allow_html=True)

    if save_as_pkl:
        with open(os.path.join(save_root, dir, 'segmentation.pkl'), 'wb+') as save_segment:
            pickle.dump(masks, save_segment)
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        13, total_steps, gradient, i), unsafe_allow_html=True)

    end_time = time.time()
    duration = end_time - start_time
    print(f'{duration}s')
    progress_placeholder.markdown(generate_segmentation_progress_bar(
        14, total_steps, gradient, i), unsafe_allow_html=True)
    # st.success(f'Segmentation of {dir}.png is done!', icon="✅")
    # st.experimental_rerun()
    if return_bbox:
        return masks, annotated_image, image_bbox
    if return_elapsed_time:
        if return_annotated:
            return masks, duration, annotated_image
        else:
            return masks, duration
    else:
        if return_annotated:
            return masks, annotated_image


def filter_out_overlapping_segmentation(sam_results):
    """
    The overlapping segments are deleted. The deletion rules are as
    follows:
        1. Compute the percentages of overlapping areas for each detected bboxes against all others, based always on the
        smaller area.
        2. If the overlapping percentages are over a specified threshold, mark the bigger bbox by setting its (x, y, _,
        ) to (0, 0, _, _)
        3. Delete the marked indices from sam_results, and return.
    :param sam_results: Raw sam_result from mask_generator.generate()
    :return: type: same format as input sam_result.
    """
    # Deepcopy cuz we don't want to mess up the original results.
    sam_result_duplicate = copy.deepcopy(sam_results)
    # Sort the sam results based on mask size, drop the first element which refers to the background
    sam_result_duplicate_sorted_based_on_mask_area = [result for result in sorted(sam_result_duplicate, key=lambda x: x[
        'area'], reverse=True)][1:]
    bboxes_from_sam = [result['bbox'] for result in sam_result_duplicate_sorted_based_on_mask_area]
    iou_matrix = np.array([[0.0] * len(bboxes_from_sam) for _ in range(len(bboxes_from_sam))])
    iou_matrix = np.array(compute_iou_list(bboxes_from_sam))

    # Set a threshold, if overlapping percentage is larger than this threshold, delete this index from sam_result
    index_ = np.where(iou_matrix > 0.85)

    # Reformat the resulting index to be more human-readable
    index_coordinated = [[index_[0][i], index_[1][i]] for i in range(len(index_[0]))]
    pop_out = mark_bigger_box(index_coordinated, bboxes_from_sam)
    final_bboxes, index_to_delete = delete_marked_bboxes_indices(pop_out)
    for index in sorted(index_to_delete, reverse=True):
        sam_result_duplicate_sorted_based_on_mask_area.pop(index)
    return sam_result_duplicate_sorted_based_on_mask_area


def is_image_very_white(masked_image, threshold=0.95):
    """
    Determines if a masked image is very white based on a specified threshold.

    Args:
        masked_image (numpy.ndarray): The masked image to evaluate.
        threshold (float, optional): The threshold to determine if the image is very white. Defaults to 0.95.

    Returns:
        bool: True if the image is very white, False otherwise.
    """
    # Convert the image to grayscale
    grayscale_image = np.mean(masked_image, axis=2)

    # Normalize the grayscale image
    normalized_image = grayscale_image / 255.0

    # # Calculate the percentage of white pixels
    white_percentage = np.mean(normalized_image >= threshold)

    # Calculate the average pixel value
    # average_pixel_value = np.mean(normalized_image)

    # Check if the image is very white
    if white_percentage >= 0.95:
        return True
    else:
        return False


def mask2box(img: np.ndarray,
             mask: np.ndarray,
             filter_white=True):
    """
    Get the bounding box of a mask and extract the bounding box from an image.
    Args:
        img: The image to apply the bounding box to.
        mask: The mask to get the bounding box from.

    Returns:
        The image with the bounding box(96x96) applied.
    """
    # Apply the mask to the image
    masked = img.copy()
    masked[~mask.astype(bool)] = [255, 255, 255]
    # Get the bounding box of the mask
    countours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find the contours
    x, y, w, h = cv2.boundingRect(countours[0])  # Get the bounding box
    # Filter the bounding box if it is too close to the edge of the image
    if x == 0 or y == 0 or x + w >= img.shape[1] - 1 or y + h >= img.shape[0] - 1:
        return None
    # Filter the bounding box if it is very white
    original_box = masked[y:y + h, x:x + w]

    if filter_white:
        if is_image_very_white(original_box):
            return None
    # Normalize the bounding box to 96x96
    w = min(w, 96)
    h = min(h, 96)
    box = np.pad(original_box, ((48 - h // 2, 48 - h // 2), (48 - w // 2, 48 - w // 2),
                                (0, 0)), mode='constant', constant_values=255)  # Pad the bounding box to 64x64
    return box


def randomized_name(dir):
    """
    Generates a randomized name for the image to be uploaded.

    Returns:
        str: The randomized name.
    """
    name = ''.join(random.choice(string.ascii_lowercase) for _ in range(10))
    if os.path.exists(os.path.join(dir, name)):
        return randomized_name() + '.png'
    else:
        return name + '.png'


def filter_and_extract_boxes(save_root, img, sam_result, dir=None, save_boxes=False):
    """
    Filters out the masks that are very white.

    Args:
        masks (list): The list of masks to filter.

    Returns:
        list: The filtered list of masks.
        list: The removed list of masks.
    """
    # Filter out the masks that are very white
    if save_boxes:
        assert dir is not None, "Please specify the directory to save boxes"
        # If boxes dir does not exist, create it
        os.makedirs(os.path.join(save_root, dir, 'boxes'), exist_ok=True)
    # print(f'Number of boxes before filtering: {len(sam_result)}')
    filtered_results = []
    removed = []
    cnt = 0
    for res in sam_result:
        mask = res['segmentation']
        mask = np.array(mask).astype(np.uint8)
        box = mask2box(img, mask)
        if box is not None:
            cnt += 1
            filtered_results.append(res)
            if save_boxes:
                name = randomized_name(
                    os.path.join(save_root, 'unlabeled'))
                cv2.imwrite(os.path.join(save_root, dir,
                                         'boxes', 'box_{}.png'.format(cnt)), box)
                cv2.imwrite(os.path.join(
                    save_root, 'unlabeled', name), box)
        else:
            removed.append(res)
    # print(f'Number of boxes: {cnt}')
    return filtered_results, removed


# def generate_head2head_comparison(dir, original_image_name, annotated, save_root):
#     desired_width, desired_height = 300, 300

#     image_bgr = cv2.imread(original_image_name)
#     # resize images
#     image_bgr = cv2.resize(image_bgr, (desired_width, desired_height))
#     annotated = cv2.resize(annotated, (desired_width, desired_height))
#     plt.figure(figsize=(10, 5))
#     sv.plot_images_grid(
#         images=[image_bgr, annotated],
#         grid_size=(1, 2),
#         titles=['source image', 'segmented image']
#     )
#     plt.savefig(os.path.join(save_root, dir,
#                              'comparison.png'))


def mark_bigger_box(bbox_indices_pairs, bboxes):
    """
    Helper function that compares the areas of bboxes with given indices, marks the bigger bbox for later deletion.
    :param bbox_indices_pairs: indices of bboxes pair. shape (2,)
    :param bboxes: List containing all detected bboxes.
    :return: Modified list containing marked bboxes.
    """
    marked_bboxes = bboxes
    for index_pair in bbox_indices_pairs:
        box_1, box_2 = index_pair
        x1, y1, w1, h1 = marked_bboxes[box_1]
        x2, y2, w2, h2 = marked_bboxes[box_2]
        box_area_1 = w1 * h1
        box_area_2 = w2 * h2

        # Compare area and mark the bigger one
        if box_area_1 > box_area_2:
            # We mark the bigger box by setting (x, y, _, _) to (0, 0, _, _)
            marked_bboxes[box_1][0], marked_bboxes[box_1][1] = 0, 0
        else:
            marked_bboxes[box_2][0], marked_bboxes[box_2][1] = 0, 0

    return marked_bboxes


def delete_marked_bboxes_indices(marked_bboxes):
    """
    Helper function that deletes the marked bboxes, must be combined with mark_bigger_bbox().
    :param marked_bboxes: List of marked indices.
    :return: final_individual_bboxes: Remaining bboxes after deletion of overlapping ones.
             index_to_delete: deleted indices, necessary for deleting the sam_result.
    """
    final_individual_bboxes = marked_bboxes
    index_to_delete = []
    for i, bbox in enumerate(marked_bboxes):
        if bbox[0] == 0 and bbox[1] == 0:
            index_to_delete.append(i)
    final_individual_bboxes = list(filter(lambda x: x[0] != 0 and x[1] != 0, final_individual_bboxes))
    return final_individual_bboxes, index_to_delete


def get_overlapping_area(bbox1, bbox2):
    """
    Compute the overlapping area between two bounding boxes, percentage based on smaller bbox.

    Arguments:
    bbox1: Tuple or list containing (x, y, w, h) of the first bounding box.
    bbox2: Tuple or list containing (x, y, w, h) of the second bounding box.

    Returns:
    overlapping_area: The overlapping area between the two bounding boxes. If the boxes do not overlap, the area is 0.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate area of each bbox
    box_area_1 = w1 * h1
    box_area_2 = w2 * h2
    box_area_smaller = min(box_area_1, box_area_2)

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Check if the boxes overlap
    if w_intersection == 0 or h_intersection == 0:
        return 0

    # Calculate the overlapping area
    overlapping_area = w_intersection * h_intersection

    # return the percentage of overlapping area based on smaller bbox
    return overlapping_area / box_area_smaller


def compute_iou_list(bboxes):
    """
    Compute IoU for every bounding box against each other in a list of bounding boxes.

    Arguments:
    bboxes: List of bounding boxes, each represented as (x, y, w, h) tuple.

    Returns:
    iou_matrix: 2D matrix containing the IoU values between all pairs of bounding boxes.
    """
    num_bboxes = len(bboxes)

    # Initialize a roi_matrix full of 0 of shape (num_bboxes * num_bboxes)
    iou_matrix = [[0.0] * num_bboxes for _ in range(num_bboxes)]

    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):
            iou = get_overlapping_area(bboxes[i], bboxes[j])
            iou_matrix[i][j] = iou
            # iou_matrix[j][i] = iou
    # Well, technically this is not IoU we are calculating, it's rather Region of Interest.
    return iou_matrix


def save_boxes_from_npy(user_folder=user_folder, dir='npy_file'):
    """
    Extracts the bounding boxes from a csv file and saves them to the specified directory.

    Args:
        dir (str): The directory to save the boxes to.
        csv_file (str): The csv file to read the masks from.
    Returns:
        None
    """
    # Filter out the masks that are very white
    for npy_file in os.listdir(os.path.join(user_folder, 'dataset', 'sam', dir)):
        img_file = npy_file.replace('.npy', '.png')
        img = cv2.imread(os.path.join(user_folder, 'dataset', 'original', img_file))
        annotations = np.load(os.path.join(
            user_folder, 'dataset', 'sam', dir, npy_file), allow_pickle=True)
        if not os.path.exists(os.path.join(user_folder, 'dataset', 'labeled')):
            os.makedirs(os.path.join(user_folder, 'dataset', 'labeled'))
        if not os.path.exists(os.path.join(user_folder, 'dataset', 'unlabeled')):
            os.makedirs(os.path.join(user_folder, 'dataset', 'unlabeled'))
        cnt = 0
        for annotation in annotations:
            mask = annotation['mask']
            label = annotation['label']
            mask = np.array(mask).astype(np.uint8)
            box = mask2box(img, mask, filter_white=False)
            if box is not None:
                cnt += 1
                if label != 'unlabeled':
                    name = label.lower() + '_' + npy_file.replace('.npy', '') + '_' + str(cnt) + '.png'
                    cv2.imwrite(os.path.join(user_folder,
                                             'dataset', 'labeled', name), box)
                else:
                    name = 'unlabeled_' + npy_file.replace('.npy', '') + '_' + str(cnt) + '.png'
                    cv2.imwrite(os.path.join(user_folder,
                                             'dataset', 'unlabeled', name), box)


def encode_image(filepath):
    """
    The image, on which the bboxes are to be annotated, must be formatted to base64.
    """
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64," + encoded


def draw_bboxes_with_widget_and_return_seg_masks(image_path, sam):
    """
    This function takes in the path of the image, on which the bboxes are to draw by user.
    Args:
        image_path: String: image path, e.g. "/.../0001.png". It takes one image once a time.
        sam: SAM model instance

    Returns:
        List: Masks of segmentation result. Can be appended to the automatic sam masks.
    """
    path = image_path
    model = sam
    mask_predictor = SamPredictor(model)
    widget = BBoxWidget()
    widget.image = encode_image(path)
    # below is the command to call in a jupyter notebook, it might not work in a .py file
    widget
    # end calling of widget
    assert widget.bboxes is not None, "You did not provide new bboxes."
    boxes = widget.bboxes

    def to_xyxy(box): return np.array([box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']])

    boxes = [to_xyxy(box) for box in boxes]

    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    masks = []
    for box in boxes:
        mask, _, _ = mask_predictor.predict(
            box=box,
            multimask_output=False
        )
        masks.append(mask)
    return masks

def draw_box_on_image(image_path, box_coordinates, color=(0, 0, 0), thickness=3):
    # Open the image
    image = Image.open(image_path)
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    # Draw the box
    draw.rectangle(box_coordinates, outline=color, width=thickness)
    # Save the modified image to the output path
    image.save(image_path)

def predict_from_bbox(image_name, boxes, save_as_pkl=True, para=parameters_segmentation_stack):
    USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
    # USER_FOLDER_PATH = os.path.join('user_folders', 'Group05')
    box_folder = os.path.splitext(os.path.basename(image_name))[0]
    masks = pickle.load(open(os.path.join(USER_FOLDER_PATH, 'dataset', 'sam', box_folder, 'segmentation.pkl'), 'rb'))
    image_path = os.path.join(USER_FOLDER_PATH, 'dataset', 'original', image_name)
    # same as image base name
    bbox_image_path = os.path.join(USER_FOLDER_PATH, 'dataset', 'sam', box_folder, 'bbox.png')
    # box_path = os.path.join(USER_FOLDER_PATH, 'dataset', 'sam', box_folder, 'output.npy')
    # boxes = np.load(box_path, allow_pickle=True)
    sam = sam_model_registry[para['MODEL_TYPE']](
    checkpoint=para['CHECKPOINT_PATH']).to(device=para['DEVICE'])
    model = sam
    mask_predictor = SamPredictor(model)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    annotated_img_path = f"{user_folder}/dataset/sam/{box_folder}/box_annotated.png"
    new_masks = []
    for box in boxes:
        mask, _, _ = mask_predictor.predict(
            box=box,
            multimask_output=False
        )
        masks.append(mask[0])
        new_masks.append(mask)
        draw_box_on_image(bbox_image_path, tuple(box))
        if os.path.exists(annotated_img_path):
            draw_box_on_image(annotated_img_path, tuple(box))
    # print("new_masks", len(new_masks))
    if save_as_pkl:
        with open(os.path.join(USER_FOLDER_PATH, 'dataset', 'sam', box_folder, 'segmentation.pkl'), 'wb') as save_segment:
            pickle.dump(masks, save_segment)
    return new_masks

# uncomment for debugging
if __name__ == '__main__':
    # boxes = np.array([[0, 0, 100, 100], [100, 100, 200, 200]])
    image_name = '0031.png'
    masks = predict_from_bbox(image_name)
    print("masks", len(masks))
    # image_path = "0011.png"
    # USER_FOLDER_PATH, SAVE_ROOT_PATH = user_folder_config()
    # mask_path, csv_path, bbox_path = get_image_masks_with_label(USER_FOLDER_PATH, image_path, 'KNN_2023_07_11_174418.joblib')
    # print(mask_path, csv_path, bbox_path)
    # abs_image_path = os.path.join(os.getcwd(), image_path)

    # print("Absolute path to image:", abs_image_path)

    # if not os.path.exists(abs_image_path):
    #     print(f"Image file not found: {abs_image_path}")
    # else:
    #     masks, duration, annotated = get_image_masks(
    #         dir, image_path, save_as_pkl=False, return_elapsed_time=True, return_annotated=True)
    #     print(duration)
    #     generate_head2head_comparison(image_path, annotated)
