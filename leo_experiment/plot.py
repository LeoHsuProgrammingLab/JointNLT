import matplotlib.pyplot as plt
from utils import read_similarity
from tqdm import tqdm
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
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
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
        print(len(all_text_img_scores), len(all_img_img_scores))

        min_frames, max_frames = 10000, 0
        plt.figure(figsize=(50, 10))
        for i, (text_img_scores, img_img_scores) in enumerate(zip(all_text_img_scores, all_img_img_scores)):
            min_frames = min(min_frames, len(text_img_scores))
            max_frames = max(max_frames, len(text_img_scores))

            mean_text_img_score = sum(text_img_scores) / len(text_img_scores)
            print(f"Mean text_img_score of video {i+1}: {mean_text_img_score}")
            
            for (text_img_score, img_img_score) in zip(text_img_scores, img_img_scores):
                plt.plot(i+1, text_img_score, 'o', alpha=0.2, color=f"C{i}", markersize=3)
        
        plt.xlabel("Video Number")
        plt.ylabel("Similarity Score")
        plt.title(f"Similarity Scores for {target}")
        plt.savefig(output_fig_dir + "all_similarities.png")
        plt.close()

        print(f"Min frames: {min_frames}, Max frames: {max_frames}")

if __name__ == "__main__":
    plot_engine = PlotEngine()
    similarity_dir = "/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/similarity/"
    output_fig_dir = "/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/figure/"
    # plot_engine.plot_similarity(similarity_dir, output_fig_dir)

    # generate video
    data_path = "/scratch/user/agenuinedream/JointNLT/data/TNL2K_test"
    # plot_engine.generate_video(data_path, type="NL")