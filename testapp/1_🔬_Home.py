import streamlit as st
from glob import glob
from streamlit_image_annotation import detection
import cv2
import numpy as np
import urllib.request
import torch
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AMI05",
    page_icon="🎯",
    # layout="wide"
)

try:
    from segment_anything import SamPredictor, sam_model_registry
except:
    import subprocess
    subprocess.call(["pip","install","git+https://github.com/facebookresearch/segment-anything.git"])
    subprocess.call(["mkdir",".sam"])
    from segment_anything import SamPredictor, sam_model_registry
    
@st.cache_data
def get_colormap(label_names, colormap_name='gist_rainbow'):
    colormap = []
    cmap = plt.get_cmap(colormap_name)
    for idx, l in enumerate(label_names):
        rgb = [int(d) for d in np.array(cmap(float(idx)/len(label_names)))*255][:3]
        #colormap[l] = ('#%02x%02x%02x' % tuple(rgb))
        colormap.append(rgb)
    return colormap
# ... rest of the code from 'sam_segmentation.py' (SAMPredictor class definition) ...
class SAMPredictor:
    def __init__(self, model_type, device, ckpt_dir='.sam'):
        os.makedirs(ckpt_dir, exist_ok=True)
        url = 'https://dl.fbaipublicfiles.com/segment_anything/'
        ckpt_dict = {'vit_b': 'sam_vit_b_01ec64.pth',
                     'vit_l': 'sam_vit_l_0b3195.pth',
                     'vit_h': 'sam_vit_h_4b8939.pth'}
        @st.cache_resource()
        def get_sam_predictor(ckpt, model_type, device):
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            return predictor
        with st.spinner('Downloading SAM model...'):
            if not os.path.exists(os.path.join(ckpt_dir, ckpt_dict[model_type])):
                urllib.request.urlretrieve(url+ckpt_dict[model_type], os.path.join(ckpt_dir, ckpt_dict[model_type]))

        self.model_type = model_type
        self.device = device
        self.predictor = get_sam_predictor(os.path.join(ckpt_dir, ckpt_dict[model_type]), model_type, device)
    
    def set_image(self, image):
        self.image = image
        self.predictor.set_image(image)

    def predict(self, bboxes):
        '''
        bboxes: [[x,y,x,y], [x,y,x,y]]
        '''
        input_boxes = torch.tensor(bboxes).to(self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, self.image.shape[:2])
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
            )
        masks = masks.cpu().numpy()
        return masks
    
    def draw_masks(self, masks, colormap, labels):
        result = self.image.copy()
        for idx, m in enumerate(masks):
            color = np.array(colormap[labels[idx]])
            h, w = m.shape[-2:]
            mask_image = m.reshape(h, w, 1) * color.reshape(1, 1, -1)
            result = np.where(np.tile(m.reshape(h, w, 1),(1,1,3)),(result * 0.5 + mask_image * 0.5).astype(np.uint8), result)
        return result
    
sam_predictor = SAMPredictor(model_type='vit_b', device='cpu')
label_list = ['WBC', 'RBC', 'AGG', 'PLT', 'OOF']
colormap = get_colormap(label_list, colormap_name='gist_rainbow')
image_path_list = glob('webapp/image/*')

if 'result_dict' not in st.session_state:
    result_dict = {}
    for img in image_path_list:
        result_dict[img] = {'bboxes': [], 'labels': [], 'masks': []}
    st.session_state['result_dict'] = result_dict.copy()

image_names = [os.path.basename(path) for path in image_path_list]
selected_image_name = st.sidebar.selectbox('Select Image', image_names, key='selectbox')
num_page = image_names.index(selected_image_name)

target_image_path = image_path_list[num_page]

new_labels = detection(image_path=target_image_path,
                       bboxes=st.session_state['result_dict'][target_image_path]['bboxes'],
                       labels=st.session_state['result_dict'][target_image_path]['labels'],
                       label_list=label_list, key=target_image_path)

if new_labels is not None:
    st.session_state['result_dict'][target_image_path]['bboxes'] = [v['bbox'] for v in new_labels]
    st.session_state['result_dict'][target_image_path]['labels'] = [v['label_id'] for v in new_labels]
    
    # Perform segmentation based on new bounding boxes
    bboxes = st.session_state['result_dict'][target_image_path]['bboxes']
    sam_predictor.set_image(cv2.imread(target_image_path))  # Assuming the image is loaded using cv2
    masks = sam_predictor.predict(bboxes)
    st.session_state['result_dict'][target_image_path]['masks'] = masks

# st.json(st.session_state['result_dict'])

image = cv2.imread(target_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sam_predictor.set_image(image)
if len(st.session_state['result_dict'][target_image_path]['bboxes'])!=0:
    input_boxes = [[b[0], b[1], b[2]+b[0], b[3]+b[1]] for b in st.session_state['result_dict'][target_image_path]['bboxes']]
    masks = sam_predictor.predict(input_boxes)
    draw_image = sam_predictor.draw_masks(masks, colormap, [l for l in st.session_state['result_dict'][target_image_path]['labels']])
    st.image(draw_image)
else:
    st.image(image)