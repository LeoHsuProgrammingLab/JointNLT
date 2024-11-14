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

def extract_llava_bbx(file_path):
    print(file_path)
    with open(file_path, "r") as file:
        lines = file.readlines()
    bbxs = []
    for line in lines:
        if '[' not in line:
            continue
        result = line.split('[')[-1].split(']')[0].strip()
    
        result = result.split(',')
        result = [coord.strip() for coord in result]
        if (len(result) < 4 or '' in result):
            bbxs.append([0, 0, 0, 0])
            continue
        bbx = [float(coord.strip()) for coord in result]
        bbxs.append(bbx)
    return bbxs

def x_y_w_h_to_x1_y1_x2_y2(bbxs):
    for bbx in bbxs:
        bbx[2] = bbx[0] + bbx[2]
        bbx[3] = bbx[1] + bbx[3]
    return bbxs

def normalized_coord_to_x1_y1_x2_y2(bbxs, width, height):
    for bbx in bbxs:
        bbx[0] = int(bbx[0] * width)
        bbx[1] = int(bbx[1] * height)
        bbx[2] = int(bbx[2] * width)
        bbx[3] = int(bbx[3] * height)
    return bbxs


def add_text_to_video(input_video_path, output_video_path, texts, position=(50, 50), font_scale=1, color=(0, 255, 0), thickness=2):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if there are no more frames
        
        # Add text to the frame
        cv2.putText(frame, texts[frame_no], position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Write the frame to the output video
        out.write(frame)
        frame_no += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")