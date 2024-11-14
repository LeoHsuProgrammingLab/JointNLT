from plot import PlotEngine
from utils import read_bbxs, extract_llava_bbx, x_y_w_h_to_x1_y1_x2_y2
import os
import cv2

def main():
    file_path = '/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/llava_text/advSamp_Baseball_game_002-Done_llava.txt'
    gt_path = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test/advSamp_Baseball_game_002-Done/groundtruth.txt'
    bbxs = extract_llava_bbx(file_path)
    bbxs_gt = read_bbx(gt_path, ',')
    bbxs_gt = x_y_w_h_to_x1_y1_x2_y2(bbxs_gt)

    if len(bbxs) < len(bbxs_gt):
        for i in range(len(bbxs_gt) - len(bbxs)):
            bbxs.append([0, 0, 0, 0])

    plot_engine = PlotEngine()
    img_dir = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test/advSamp_Baseball_game_002-Done/imgs'
    output_video_path = '/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/llava_video/advSamp_Baseball_game_002-Done_llava.mp4'

    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    imgs = sorted(imgs)

    first_frame = cv2.imread(imgs[0])
    if first_frame is None:
        print(f"Error: Unable to read the first image {imgs[0]}.")
        return

    height, width, layers = first_frame.shape
    fps = 30

    # modfify normalized bbxs to original size
    for bbx in bbxs:
        bbx[0] = int(bbx[0] * width)
        bbx[1] = int(bbx[1] * height)
        bbx[2] = int(bbx[2] * width)
        bbx[3] = int(bbx[3] * height)
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Draw bounding boxes and write video frames
    plot_engine.draw_bbxs(imgs, bbxs, bbxs_gt, "", video_writer)
    
    # Release video writer
    video_writer.release()
    print(f"Video saved successfully at {output_video_path}")

if __name__ == "__main__":
    main()