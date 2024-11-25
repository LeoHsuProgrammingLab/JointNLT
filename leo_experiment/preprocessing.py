from plot import PlotEngine
from utils import read_bbx, extract_llava_bbx, x_y_w_h_to_x1_y1_x2_y2
import os
import cv2
import shutil
from tqdm import tqdm

def read_dir_files(dir_path, ext='.mp4'):
    return [os.path.splitext(f)[0] for f in os.listdir(dir_path) if f.endswith(ext)]

def copy_dirs(src_dir, dest_dir, dir_names):
    for dir_name in tqdm(dir_names):
        src_path = os.path.join(src_dir, dir_name)
        dest_path = os.path.join(dest_dir, dir_name)

        if (os.path.exists(dest_path)):
            shutil.rmtree(dest_path)
        
        shutil.copytree(src_path, dest_path)

def construct_a_subset():
    dir_path = '/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/video_subset'
    dir_names = read_dir_files(dir_path)
    src_dir = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test_all'
    dest_dir = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test'
    if (not os.path.exists(dest_dir)):
        os.makedirs(dest_dir)
    copy_dirs(src_dir, dest_dir, dir_names)

def llava_bbx_to_video(gt_dir, llava_dir, output_dir):
    file_names = os.listdir(gt_dir)
    file_names.sort()
    for file_name in file_names:
        llava_file = os.path.join(llava_dir, f"{file_name}_llava.txt")
        gt_file = os.path.join(gt_dir, f"{file_name}/groundtruth.txt")
        frame_num = len(os.listdir(os.path.join(gt_dir, file_name, 'imgs')))

        bbxs, refined_texts = extract_llava_bbx(llava_file, frame_num)
        bbxs_gt = read_bbx(gt_file, ',')
        bbxs_gt = x_y_w_h_to_x1_y1_x2_y2(bbxs_gt)

        assert(len(bbxs) == len(bbxs_gt))

        print(f"Processing {file_name}...")
        
        # write the bbxs & refined_texts to output_dir
        output_text_dir = os.path.join(output_dir, 'llava_text_subset')
        output_bbx_dir = os.path.join(output_dir, 'llava_bbx_subset')
        if (not os.path.exists(output_text_dir)):
            os.makedirs(output_text_dir)
        if (not os.path.exists(output_bbx_dir)):
            os.makedirs(output_bbx_dir)

        # write the texts
        with open(os.path.join(output_text_dir, f"{file_name}.txt"), 'w') as f:
            for text in refined_texts:
                f.write(text + '\n')

        # write the video
        output_video_dir = os.path.join(output_dir, 'llava_video_subset')
        output_video_path = os.path.join(output_video_dir, f"{file_name}.mp4")
        plot_engine = PlotEngine()

        img_dir = os.path.join(gt_dir, file_name, 'imgs')
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

        # write the bbxs
        with open(os.path.join(output_bbx_dir, f"{file_name}.txt"), 'w') as f:
            for bbx in bbxs:
                f.write(f"{bbx[0]},{bbx[1]},{bbx[2]},{bbx[3]}\n")
    
        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
        # Draw bounding boxes and write video frames
        plot_engine.draw_bbxs_and_texts(imgs, bbxs, bbxs_gt, refined_texts, video_writer)
    
        # Release video writer
        video_writer.release()
        print(f"Video saved successfully at {output_video_path}")

def main():
    # construct_a_subset()
    gt_dir = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test'
    llava_dir = '/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/llava_text'
    output_dir = '/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track'

    # llava_bbx_to_video(gt_dir, llava_dir, output_dir)

    llava_text_subset_dir = os.path.join(output_dir, 'llava_text_subset')
    llava_texts = os.listdir(llava_text_subset_dir)
    llava_texts.sort()

    for llava_text in tqdm(llava_texts):
        with open(os.path.join(llava_text_subset_dir, llava_text), 'r') as f:
            # remove the empty lines
            texts = [line.strip() for line in f if line.strip()]
            # remove the content of file and write the texts to the same file
            with open(os.path.join(llava_text_subset_dir, llava_text), 'w') as f:
                for text in texts:
                    f.write(text + '\n')

if __name__ == "__main__":
    main()