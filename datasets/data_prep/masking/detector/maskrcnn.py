import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class Segmentation:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self._initialize_mrcnn_model()

    def _initialize_mrcnn_model(self):
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        self.model = model.to(self.device)

    def get_prediction(self, img, threshold=0.6):
        """
        get_prediction
        parameters:
        - img_path - path of the input image
        method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
            ie: eg. segment of cat is made 1 and rest of the image is made 0

        """
        #img = Image.open(img_path)
        #img = cv2.imread(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img).unsqueeze(0)
        pred = self.model(img.to(self.device))
        pred_label = pred[0]['labels']
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = len([pred_score.index(x) for x in pred_score if x > threshold]) - 1
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = np.array([COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().numpy())])
        pred_confs = np.array(pred_score)
        pred_boxes = np.array([[(i[0] + i[2]) / 2, (i[1] + i[3]) / 2, i[2] - i[0], i[3] - i[1]] for i in list(pred[0]['boxes'].detach().cpu().numpy())])
        pred_label = pred_label[:pred_t+1].detach().cpu().numpy()

        label_mask = [(pred_label == 1) | (pred_label == 3) | (pred_label == 4) | (pred_label == 6) | (pred_label == 8)]

        if len(masks.shape) < 3:
            return np.expand_dims(masks, axis=0), pred_boxes, pred_confs, pred_class
        else:
            masks = masks[:pred_t+1][label_mask]
            pred_boxes = pred_boxes[:pred_t+1][label_mask]
            pred_confs = pred_confs[:pred_t+1][label_mask]
            pred_class = pred_class[:pred_t+1][label_mask]

            return masks, pred_boxes, pred_confs, pred_class
