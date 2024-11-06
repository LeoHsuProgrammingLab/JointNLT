import cv2
import os
from tqdm.auto import tqdm

def read_bbx(bbx_txt_path, sep=""):
    bbxs = []
    with open(bbx_txt_path, 'r') as f:
        for line in f:
            if sep == "":
                coords = list(map(int, line.strip().split()))
            else:
                coords = list(map(int, line.strip().split(sep)))
            bbxs.append(coords)
    
    return bbxs

def read_text(text_path):
    text = ""
    with open(text_path, 'r') as f:
        for line in f:
            text += line + '\n'
    
    return text

def read_similarity(similarity_path, target='Template'):
    text_img_score = []
    img_img_score = []
    with open(similarity_path, 'r') as f:
        for idx, line in enumerate(f):
            # skip first line because first line is the specification of the file
            if idx == 0:
                continue

            if target == 'Template':
                text_img_score.append(float(line.split(",")[0].split(":")[1].strip()))
                img_img_score.append(float(line.split(",")[1].split(":")[1].strip()))
            elif target == 'Frame':
                text_img_score.append(float(line.split(",")[2].split(":")[1].strip()))
                img_img_score.append(float(line.split(",")[3].split(":")[1].strip()))
            else:
                raise ValueError(f"Invalid target: {target}")
    
    return text_img_score, img_img_score

def read_iou(iou_path):
    iou_scores = []
    with open(iou_path, 'r') as f:
        for line in f:
            iou_scores.append(float(line.strip()))
    
    return iou_scores