import os
import torch
import open_clip
from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import Image
from utils import read_text, read_bbx
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

class ExperimentEngine:
    def __init__(self, model_name="ViT-B/32"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
        self.text = None
        self.text_features = None
        self.template = None
        self.template_features = None
        self.first_frame = None
        self.first_frame_features = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def model_eval(self):
        self.model.eval()
        self.model.to(self.device)

    def get_similarity(self, feature_1, feature_2):
        return (feature_1 @ feature_2.T).item()

    # Similarity Calculation
    def calculate_similarity(self, img_path, bbox_predicted=None, bbox_base=None):
        if (bbox_predicted == [0, 0, 0, 0] 
            or bbox_predicted[0] == bbox_predicted[2] 
            or bbox_predicted[1] == bbox_predicted[3]
        ):
            return 0, 0
        
        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")

        # predicted bounding box
        if bbox_predicted is not None:
            target = image.crop(bbox_predicted)
        else:
            target = image
        
        target_preprocessed = self.preprocess(target).unsqueeze(0).to(self.device)
        
        # Get CLIP embeddings
        with torch.no_grad(): # torch.cuda.amp.autocast() when need to reduce memory usage
            target_features = self.model.encode_image(target_preprocessed).to(self.device)

        # Calculate CLIPScores (cosine similarity) between each ROI and the text
        text_img_score = self.get_similarity(self.text_features, target_features)
        assert text_img_score == F.cosine_similarity(self.text_features, target_features).item(), "Mismatch between similarity calculation methods."
        img_img_score = self.get_similarity(self.first_frame_features if bbox_predicted is None else self.template_features, target_features)
        assert img_img_score == F.cosine_similarity(self.first_frame_features if bbox_predicted is None else self.template_features, target_features).item(), "Mismatch between similarity calculation methods."
  
        return text_img_score, img_img_score

    def run_similarity(self, img_dir, text_dir, pred_dir, output_path, random=False):
        img_names = sorted([img_dir + img for img in os.listdir(img_dir) if img.endswith((".jpg", ".png"))])
        
        try: # (x, y, w, h)
            bbxs_original = read_bbx(pred_dir) # predicted bounding boxes
        except:
            bbxs_original = read_bbx(pred_dir, ',') # gt bounding boxes
        
        bbxs = [[bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]] for bbx in bbxs_original] # (x1, y1, x2, y2)
        
        # Text
        self.text = read_text(text_dir)

        # First Frame
        self.first_frame = Image.open(img_names[0]).convert("RGB")
        self.first_frame_features = self.model.encode_image(self.preprocess(self.first_frame).unsqueeze(0).to(self.device))
        self.first_frame_features /= self.first_frame_features.norm(dim=-1, keepdim=True)
        
        # Template
        self.template = self.first_frame.crop(bbxs[0])
        template_preprocessed = self.preprocess(self.template).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.template_features = self.model.encode_image(template_preprocessed).to(self.device)
            self.template_features /= self.template_features.norm(dim=-1, keepdim=True)

            self.text_features = self.model.encode_text(self.tokenizer(self.text).to(self.device))
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        assert len(bbxs) == len(img_names), "Mismatch between bounding boxes and images."

        with open(output_path, "w") as f:
            for i, img_name in enumerate(img_names):
                if random:
                    random_crop_bbox = self.generate_random_bbox(self.first_frame.size, bbxs_original[i])
                    text_visual_score, template_visual_score = self.calculate_similarity(img_name, random_crop_bbox)
                else:
                    text_visual_score, template_visual_score = self.calculate_similarity(img_name, bbxs[i])

                f.write(f"Text_PredBBX: {round(text_visual_score, 3)}, Template_PredBBX: {round(template_visual_score, 3)}\n")
                # text_frame_score, frame_first_frame_score = self.calculate_similarity(img_name)
                # if (i == 0):
                #     f.write(f"text and template similarity = {round(text_visual_score, 3)}, text and first frame similarity = {round(text_frame_score, 3)}, \n")                    
                
                # f.write(f"Text_PredBBX: {round(text_visual_score, 3)}, Template_PredBBX: {round(visual_template_score, 3)}, Text_Frame: {round(text_frame_score, 3)}, Frame_FirstFrame: {round(frame_first_frame_score, 3)}\n")
    
    def generate_random_bbox(self, frame_size, bbx):
        # random crop a bounding box in the image: (x, y, w, h), must be in the frame size
        crop_x = np.random.randint(0, frame_size[0] - bbx[2]) if frame_size[0] > bbx[2] else 0
        crop_y = np.random.randint(0, frame_size[1] - bbx[3]) if frame_size[1] > bbx[3] else 0
        crop_w = bbx[2]
        crop_h = bbx[3]
        return (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)

    def calculate_iou(self, bbx1, bbx2):
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        x1_min, y1_min, x1_max, y1_max = bbx1[0], bbx1[1], bbx1[0] + bbx1[2], bbx1[1] + bbx1[3]
        x2_min, y2_min, x2_max, y2_max = bbx2[0], bbx2[1], bbx2[0] + bbx2[2], bbx2[1] + bbx2[3]

        # Coordinates of the intersection rectangle
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Area of both bounding boxes
        bbx1_area = bbx1[2] * bbx1[3]
        bbx2_area = bbx2[2] * bbx2[3]

        # Union area
        union_area = bbx1_area + bbx2_area - intersection_area

        # IoU
        iou = intersection_area / union_area
        return iou

    def run_iou(self, ground_truth_path, pred_path, output_path):
        ground_truth = read_bbx(ground_truth_path, ',')
        pred = read_bbx(pred_path)

        assert len(ground_truth) == len(pred), "Mismatch between ground truth and predictions."

        with open(output_path, "w") as f:
            for i in range(len(ground_truth)):
                iou = self.calculate_iou(ground_truth[i], pred[i])
                f.write(f"{round(iou, 3)}\n")

def main():
    def parse_args():
        parser = argparse.ArgumentParser(description="Run similarity and IoU calculations for image and text data.")
        parser.add_argument("--data_path", type=str, default="/scratch/user/agenuinedream/JointNLT/data/TNL2K_test", help="Path to the dataset.")
        parser.add_argument("--pred_path", type=str, default="/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track", help="Path to the prediction results.")
        parser.add_argument("--output_dir_name", type=str, default="gt_sim", help="Name of the output directory.")
        return parser.parse_args()

    args = parse_args()
    engine = ExperimentEngine()
    data_path = args.data_path
    subset_names = [subset_name for subset_name in os.listdir(data_path)]
    subset_names.sort()

    pred_path = args.pred_path
    output_dir_name = args.output_dir_name

    if not os.path.exists(f"{pred_path}/{output_dir_name}"):
        os.makedirs(f"{pred_path}/{output_dir_name}")

    engine.model_eval()
    for i, subset_name in tqdm(enumerate(subset_names), total=len(subset_names)):
        img_dir = f"{data_path}/{subset_name}/imgs/"
        text_file = f"{data_path}/{subset_name}/language.txt"
        # pred_file = f"{pred_path}/{subset_name}.txt"
        gt_file = f"{data_path}/{subset_name}/groundtruth.txt"
        output_path = f"{pred_path}/gt_sim/{subset_name}.txt"

        engine.run_similarity(img_dir, text_file, gt_file, output_path)

        # Calculate IoU
        # output_path = f"{pred_path}/iou/{subset_name}.txt"
        # engine.run_iou(gt_file, pred_file, output_path)

if __name__ == "__main__":
    main()

