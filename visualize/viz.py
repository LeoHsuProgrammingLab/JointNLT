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

def draw_bbxs(imgs, bbxs, bbxs_gt, text, video_writer):
    # Iterate through the images and bounding boxes
    for i, img_file in enumerate(imgs):
        # Read the image
        frame = cv2.imread(img_file)
        
        # Get the bounding box for this frame
        x_min, y_min, w, h = bbxs[i]
        x_max, y_max = x_min + w, y_min + h
        x_min_gt, y_min_gt, w_gt, h_gt = bbxs_gt[i]
        x_max_gt, y_max_gt = x_min_gt + w_gt, y_min_gt + h_gt

        # print(f"Predicted bbox: {(x_min, y_min, x_max, y_max)}")
        # print(f"Ground truth bbox: {(x_min_gt, y_min_gt, x_max_gt, y_max_gt)}")
        
        # Draw the bounding box on the image (color is blue, thickness is 2)
        # [x1, y1, w, h]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.rectangle(frame, (x_min_gt, y_min_gt), (x_max_gt, y_max_gt), (0, 0, 255), 2)

        # text
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        # Write the frame with the bounding box to the video
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()

def generate_video(subset_name):
    try:
        # Define paths
        bbx_txt_path = f"../test/tracking_results/jointnlt/swin_b_ep300_track/{subset_name}.txt"
        bbx_gt_txt_path = f"../data/TNL2K_test/{subset_name}/groundtruth.txt"
        text_path = f"../data/TNL2K_test/{subset_name}/language.txt"
        imgs_dir_path = f"../data/TNL2K_test/{subset_name}/imgs/"
        output_video_path = f"../test/tracking_results/jointnlt/swin_b_ep300_track/{subset_name}.mp4"
        # if os.path.exists(output_video_path):
        #     print(f"{subset_name} video already generated")
        #     return
        
        # Check if the paths exist
        if not os.path.exists(bbx_txt_path):
            print(f"Error: Bounding box file {bbx_txt_path} does not exist.")
            return
        if not os.path.exists(bbx_gt_txt_path):
            print(f"Error: Ground truth bounding box file {bbx_gt_txt_path} does not exist.")
            return
        if not os.path.exists(text_path):
            print(f"Error: Language file {text_path} does not exist.")
            return
        if not os.path.isdir(imgs_dir_path):
            print(f"Error: Image directory {imgs_dir_path} does not exist.")
            return

        # sort images
        imgs = [imgs_dir_path + img for img in os.listdir(imgs_dir_path) if img.endswith((".jpg", ".png"))]
        if not imgs:
            print(f"Error: No images found in directory {imgs_dir_path}.")
            return
        imgs.sort()

        # get bbxs
        bbxs = read_bbx(bbx_txt_path)
        bbxs_gt = read_bbx(bbx_gt_txt_path, ",")

        # get text
        text = read_text(text_path)

        assert len(bbxs) == len(imgs), "Mismatch between bounding boxes and images."

        # Read the first image to get dimensions
        first_frame = cv2.imread(imgs[0])
        if first_frame is None:
            print(f"Error: Unable to read the first image {imgs[0]}.")
            return
        height, width, layers = first_frame.shape
        fps = 30

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Draw bounding boxes and write video frames
        draw_bbxs(imgs, bbxs, bbxs_gt, text, video_writer)

        # Release video writer
        video_writer.release()
        print(f"Video saved successfully at {output_video_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return

if __name__ == "__main__":
    data_path = "/scratch/user/agenuinedream/JointNLT/data/TNL2K_test"
    subset_names = [subset_name for subset_name in os.listdir(data_path)]
    subset_names.sort()

    for name in tqdm(subset_names):
        generate_video(name)