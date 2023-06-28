import streamlit as st
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
import cv2
import os
import time
import supervision as sv
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


parameters_segmentation_stack = {
    'CHECKPOINT_PATH': os.path.join("./", "sam_weights", "sam_vit_b_01ec64.pth"),
    'MODEL_TYPE': "vit_b",
    'DEVICE': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}

SAVE_ROOT_PATH = "data"
# Description:
# you need to locate to your main project folder,
# where you need a folder named: sam_weights, for storing the pretrained weights.
# The pretrained weights can be found in the group_share folder


def generate_progress_bar(count, total_count, gradient, idx):
    display_count = count / total_count * 100
    css = f"""
    <style>
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


def get_image_masks(i,dir, image_name, save_as_pkl=False, save_annotated=False, return_elapsed_time=False, return_annotated=False, para=parameters_segmentation_stack):
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
    progress_placeholder.markdown(generate_progress_bar(
        1, total_steps, gradient,i), unsafe_allow_html=True)

    mask_generator = SamAutomaticMaskGenerator(sam)
    progress_placeholder.markdown(generate_progress_bar(
        2, total_steps, gradient,i), unsafe_allow_html=True)

    assert os.path.exists(os.path.join(SAVE_ROOT_PATH, dir)
                          ), "Directory does not exist"
    progress_placeholder.markdown(generate_progress_bar(
        3, total_steps, gradient,i), unsafe_allow_html=True)
    
    image_bgr = cv2.imread(image_name)
    time.sleep(0.1)
    progress_placeholder.markdown(generate_progress_bar(
        10, total_steps, gradient,i), unsafe_allow_html=True)
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)

    sam_result = mask_generator.generate(image_rgb)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)

    if len(sam_result) > 1:
        sam_result = filter_and_extract_boxes(
            image_bgr, sam_result, dir=dir, save_boxes=True)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)
    
    masks = [mask['segmentation'] for mask in sorted(
        sam_result, key=lambda x: x['area'], reverse=True)][1:]
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)

    mask_annotator = sv.MaskAnnotator()
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)
    
    detections = sv.Detections.from_sam(sam_result=sam_result)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)
    
    annotated_image = mask_annotator.annotate(
        scene=image_bgr.copy(), detections=detections)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)
    box_annotator = sv.BoxAnnotator()
    image_bbox = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
    if save_annotated:
        cv2.imwrite(os.path.join(SAVE_ROOT_PATH, dir,
                    'segmented.png'), annotated_image)
        cv2.imwrite(os.path.join(SAVE_ROOT_PATH, dir, 'bbox.png'), image_bbox)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)

    # generate_head2head_comparison(dir, image_name, annotated_image)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)

    if save_as_pkl:
        with open(os.path.join(SAVE_ROOT_PATH, dir, 'segmentation.pkl'), 'wb') as save_segment:
            pickle.dump(masks, save_segment)
    progress_placeholder.markdown(generate_progress_bar(
        13, total_steps, gradient,i), unsafe_allow_html=True)

    end_time = time.time()
    duration = end_time-start_time
    print(f'{duration}s')
    progress_placeholder.markdown(generate_progress_bar(
        14, total_steps, gradient,i), unsafe_allow_html=True)
    # st.success(f'Segmentation of {dir}.png is done!', icon="✅")
    # st.experimental_rerun()

    if return_elapsed_time:
        if return_annotated:
            return masks, duration, annotated_image
        else:
            return masks, duration
    else:
        if return_annotated:
            return masks, annotated


def is_image_very_white(masked_image, threshold=0.92):
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

    # Calculate the percentage of white pixels
    white_percentage = np.mean(normalized_image >= threshold)

    # Check if the image is very white
    if white_percentage >= threshold:
        return True
    else:
        return False


def mask2box(img: np.ndarray,
             mask: np.ndarray):
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
    if x == 0 or y == 0 or x + w >= img.shape[1]-1 or y + h >= img.shape[0]-1:
        return None
    # Filter the bounding box if it is very white
    original_box = img[y:y+h, x:x+w]
    if is_image_very_white(original_box):
        return None
    # Normalize the bounding box to 96x96
    w = min(w, 96)
    h = min(h, 96)
    box = np.pad(original_box, ((48 - h // 2, 48 - h // 2), (48 - w // 2, 48 - w // 2),
                 (0, 0)), mode='constant', constant_values=255)  # Pad the bounding box to 64x64
    return box


def filter_and_extract_boxes(img, sam_result, dir=None, save_boxes=False):
    """
    Filters out the masks that are very white.

    Args:
        masks (list): The list of masks to filter.

    Returns:
        list: The filtered list of masks.
    """
    # Filter out the masks that are very white
    if save_boxes:
        assert dir is not None, "Please specify the directory to save boxes"
        # If boxes dir does not exist, create it
        os.makedirs(os.path.join(SAVE_ROOT_PATH, dir, 'boxes'), exist_ok=True)
    filtered_results = []
    cnt = 0
    for res in sam_result:
        mask = res['segmentation']
        mask = np.array(mask).astype(np.uint8)
        box = mask2box(img, mask)
        if box is not None:
            cnt += 1
            filtered_results.append(res)
            if save_boxes:
                cv2.imwrite(os.path.join(SAVE_ROOT_PATH, dir,
                            'boxes', 'box_{}.png'.format(cnt)), box)
    return filtered_results


def generate_head2head_comparison(dir, original_image_name, annotated):
    desired_width, desired_height = 300, 300 

    image_bgr = cv2.imread(original_image_name)
    # resize images
    image_bgr = cv2.resize(image_bgr, (desired_width, desired_height))
    annotated = cv2.resize(annotated, (desired_width, desired_height))
    plt.figure(figsize=(10, 5))
    sv.plot_images_grid(
        images=[image_bgr, annotated],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )
    plt.savefig(os.path.join(SAVE_ROOT_PATH, dir,
                             'comparison.png'))


# uncomment for debugging
# if __name__ == '__main__':
#     image_path = "app/image/0034.png"
#     abs_image_path = os.path.join(os.getcwd(), image_path)

#     print("Absolute path to image:", abs_image_path)

#     if not os.path.exists(abs_image_path):
#         print(f"Image file not found: {abs_image_path}")
#     else:
#         masks, duration, annotated = get_image_masks(
#             dir, image_path, save_as_pkl=False, return_elapsed_time=True, return_annotated=True)
#         print(duration)
#         generate_head2head_comparison(image_path, annotated)
