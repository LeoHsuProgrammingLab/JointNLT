import matplotlib.pyplot as plt
from utils import read_similarity, read_iou, read_bbx, read_text
from tqdm import tqdm
from data import Normalizer
import os

class PlotEngine():
    def __init__(self):
        pass

    def draw_bbxs(self, imgs, bbxs, bbxs_gt, text, video_writer):
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
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            
            # Write the frame with the bounding box to the video
            video_writer.write(frame)
        
        # Release the video writer
        video_writer.release()

    def generate_video(data_path, type="NL_BB"):
        subset_names = [subset_name for subset_name in os.listdir(data_path)]
        subset_names = sorted(subset_names)

        target = "swin_b_ep300_track" if type == 'NL_BB' else "swin_b_ep300"

        for i, subset_name in tqdm(enumerate(subset_names), total=len(subset_names)):
            try:
                # Define paths
                bbx_txt_path = f"../test/tracking_results/jointnlt/{target}/{subset_name}.txt"
                output_video_path = f"../test/tracking_results/jointnlt/{target}/{subset_name}.mp4"

                # Define subset paths
                bbx_gt_txt_path = f"../data/TNL2K_test/{subset_name}/groundtruth.txt"
                text_path = f"../data/TNL2K_test/{subset_name}/language.txt"
                imgs_dir_path = f"../data/TNL2K_test/{subset_name}/imgs/"
                
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

    def plot_similarity(self, similarity_dir, output_fig_dir):
        similarity_files = sorted([similarity_dir + sim for sim in os.listdir(similarity_dir) if sim.endswith(".txt")])
        target = "Template"

        # Plot the similarity scores
        all_text_img_scores = []
        all_img_img_scores = []
        for file in similarity_files:
            text_img_scores, img_img_scores = read_similarity(file, target=target)
            all_text_img_scores.append(text_img_scores)
            all_img_img_scores.append(img_img_scores)

        min_frames, max_frames = 10000, 0
        plt.figure(figsize=(20, 10))
        for i, (text_img_scores, img_img_scores) in tqdm(enumerate(zip(all_text_img_scores, all_img_img_scores)), total=len(all_text_img_scores)):
            for (text_img_score, img_img_score) in zip(text_img_scores, img_img_scores):
                plt.plot(i+1, text_img_score, 'o', alpha=0.2, color=f"C{i}", markersize=2)

            if (i%200 == 199 or i == len(all_text_img_scores)-1):
                print(f"plot!")
                plt.xlabel("Video Number")
                plt.ylabel("Similarity Score")
                plt.title(f"Similarity Scores for {target}")
                plt.savefig(output_fig_dir + f"all_similarities_{i//200 + 1}.png")
                plt.clf()
        
        plt.close()
    
    def plot_iou(self, iou_dir, output_fig_dir):
        iou_files = sorted([iou_dir + iou for iou in os.listdir(iou_dir) if iou.endswith(".txt")])

        # Plot the IoU scores
        all_iou_scores = []
        for file in iou_files:
            iou_scores = read_iou(file)
            all_iou_scores.append(iou_scores)

        plt.figure(figsize=(20, 10))
        for i, iou_scores in tqdm(enumerate(all_iou_scores), total=len(all_iou_scores)):
            mean_iou = sum(iou_scores) / len(iou_scores)

            plt.plot(i+1, mean_iou, 'o', alpha=1, color=f"C{i}", markersize=3)

            if (i%200 == 199 or i == len(all_iou_scores)-1):
                print(f"plot!")
                plt.xlabel("Video Number")
                plt.ylabel("IoU Score")
                plt.title("Mean IoU Scores")
                plt.savefig(output_fig_dir + f"all_iou_scores_{i//200 + 1}.png")
                plt.clf()
        
        plt.close()

    def plot_similarity_iou(self, similarity_dir, iou_dir, output_fig_dir):
        similarity_files = sorted([similarity_dir + sim for sim in os.listdir(similarity_dir) if sim.endswith(".txt")])
        iou_files = sorted([iou_dir + iou for iou in os.listdir(iou_dir) if iou.endswith(".txt")])

        for similarity_file, iou_file in tqdm(zip(similarity_files, iou_files), total=len(similarity_files)):
            text_img_scores, img_img_scores = read_similarity(similarity_file)
            iou_scores = read_iou(iou_file)

            # text_img_scores = Normalizer().min_max_normalize(text_img_scores)
            # img_img_scores = Normalizer().min_max_normalize(img_img_scores)
            # iou_scores = Normalizer().min_max_normalize(iou_scores)

            plt.figure(figsize=(15, 5))
            plt.plot(range(1, len(text_img_scores)+1), text_img_scores, label="Text-Image", alpha=0.7)
            plt.plot(range(1, len(img_img_scores)+1), img_img_scores, label="Image-Image", alpha=0.7)
            plt.plot(range(1, len(iou_scores)+1), iou_scores, label="IoU", alpha=0.7)
            plt.xlabel("Frame Number")
            plt.ylabel("Score")
            plt.title(f"Similarity Scores and IoU")
            plt.legend()
            plt.savefig(output_fig_dir + f"{similarity_file.split('/')[-1].split('.')[0]}.png")
            plt.close()
        
def main():
    plot_engine = PlotEngine()
    similarity_dir = "/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/gt_sim/"
    iou_dir = "/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/iou/"
    output_fig_dir = "/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/figure/sim_gt_iou/"
    if not os.path.exists(output_fig_dir):
        os.makedirs(output_fig_dir)
    plot_engine.plot_similarity_iou(similarity_dir, iou_dir, output_fig_dir)
    # plot_engine.plot_similarity(similarity_dir, output_fig_dir)
    # plot_engine.plot_iou(iou_dir, output_fig_dir)

    # generate video
    data_path = "/scratch/user/agenuinedream/JointNLT/data/TNL2K_test"
    # plot_engine.generate_video(data_path, type="NL")

if __name__ == "__main__":
    main()