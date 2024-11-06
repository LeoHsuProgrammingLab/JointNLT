import os
import numpy as np
from tqdm import tqdm
from utils import read_similarity, read_iou

class EvaluationEngine:
    def __init__(self):
        pass
    
    def get_min_max_similarity(self, similarity_dir):
        similarity_files = sorted([similarity_dir + file for file in os.listdir(similarity_dir) if file.endswith(".txt")])
        min_sim_img_img, min_sim_text_img = 1, 1
        max_sim_img_img, max_sim_text_img = 0, 0
        for similarity_file in tqdm(similarity_files):
            text_img_scores, img_img_scores = read_similarity(similarity_file, target="Frame")
            min_sim_text_img = min(min_sim_text_img, min(text_img_scores))
            min_sim_img_img = min(min_sim_img_img, min(img_img_scores))
            max_sim_text_img = max(max_sim_text_img, max(text_img_scores))
            max_sim_img_img = max(max_sim_img_img, max(img_img_scores))
        print(f"Min similarity between text-image: {min_sim_text_img}, Max similarity between text-image: {max_sim_text_img}")
        print(f"Min similarity between image-image: {min_sim_img_img}, Max similarity between image-image: {max_sim_img_img}")

    def sim_iou_correlation(self, simimalrity_list, iou_list):
        assert len(simimalrity_list) == len(iou_list), "Mismatch between similarity and IoU lists."
        return np.corrcoef(simimalrity_list, iou_list)[0, 1]
    
    def eval_sim_iou_correlation(self, similarity_dir, iou_dir):
        similarity_files = sorted([similarity_dir + file for file in os.listdir(similarity_dir) if file.endswith(".txt")])
        iou_files = sorted([iou_dir + file for file in os.listdir(iou_dir) if file.endswith(".txt")])

        assert len(similarity_files) == len(iou_files), "Mismatch between similarity and IoU files."

        for similarity_file, iou_file in tqdm(zip(similarity_files, iou_files), total=len(similarity_files)):
            text_img_scores, img_img_scores = read_similarity(similarity_file)
            iou_scores = read_iou(iou_file)

            correlation = self.sim_iou_correlation(text_img_scores, iou_scores)
            print(f"Correlation between text-image similarity and IoU: {correlation}")

            correlation = self.sim_iou_correlation(img_img_scores, iou_scores)
            print(f"Correlation between image-image similarity and IoU: {correlation}")

def main():
    eval_engine = EvaluationEngine()
    similarity_dir = "/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/similarity/"
    iou_dir = "/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/iou/"
    # eval_engine.eval_sim_iou_correlation(similarity_dir, iou_dir)
    eval_engine.get_min_max_similarity(similarity_dir)

if __name__ == "__main__":
    main()