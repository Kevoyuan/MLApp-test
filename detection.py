import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
import cv2
import os
import time
import supervision as sv
import urllib.request


# Description:
# you need to locate to your main project folder,
# where you need a folder named: sam_weights, for storing the pretrained weights.
# The pretrained weights can be found in the group_share folder
os.chdir('../')
HOME = os.getcwd()

# The pretrained weights are stored in the sam_weights/ folder
parameters_segmentation_stack = {
    'CHECKPOINT_PATH': os.path.join(HOME, "sam_weights", "sam_vit_b_01ec64.pth"),
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
    print(f"Downloading {model_type} model...")
    urllib.request.urlretrieve(url + ckpt_file, os.path.join(ckpt_dir, ckpt_file))
    print("Download complete.")
else:
    print("Model exists.")

def get_image_masks(image_name='0000.png', save_as_pkl=False, return_elapsed_time=False, return_annotated=False ,para=parameters_segmentation_stack):
    """
    returns masks of given image
    :param image_name: name of the iamge
    :param save_as_pkl: a flag
    :param model: default to sam
    :return: masks, original or as pickled file
    """
    st = time.time()
    sam = sam_model_registry[para['MODEL_TYPE']](checkpoint=para['CHECKPOINT_PATH']).to(device=para['DEVICE'])
    mask_generator = SamAutomaticMaskGenerator(sam)
    image_bgr = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)
    masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)][1:]
    mask_annotator = sv.MaskAnnotator()
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    if save_as_pkl:
        with open(image_name.removesuffix('.png') + '.pkl', 'wb') as save_segment:
            pickle.dump(masks, save_segment)
    et = time.time()
    duration = et-st
    if return_elapsed_time:
        if return_annotated:
            return masks, duration, annotated_image
        else:
            return masks, duration
    else:
        if return_annotated:
            return masks, annotated_image

def generate_head2head_comparison(original_image_name, annotated):
    image_bgr = cv2.imread(original_image_name)
    sv.plot_images_grid(
        images=[image_bgr, annotated],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

# uncomment for debugging
if __name__ == '__main__':
    image_path = "./app/image/0000.png"
    abs_image_path = os.path.join(os.getcwd(), image_path)
    
    print("Absolute path to image:", abs_image_path)

    if not os.path.exists(abs_image_path):
        print(f"Image file not found: {abs_image_path}")
    else:
        masks, duration, annotated = get_image_masks(image_path, save_as_pkl=False, return_elapsed_time=True, return_annotated=True)
        print(duration)
        generate_head2head_comparison(image_path, annotated)
