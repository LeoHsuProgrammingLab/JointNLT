import cv2
import os

def read_bbx(bbx_txt_path):
    bbxs = []
    with open(bbx_txt_path, 'r') as f:
        for line in f:
            coords = list(map(int, line.strip().split()))
            bbxs.append(coords)
    
    return bbxs

def draw_bbxs(imgs, bbxs, video_writer):
    # Iterate through the images and bounding boxes
    for i, img_file in enumerate(imgs):
        # Read the image
        frame = cv2.imread(img_file)
        
        # Get the bounding box for this frame
        x_min, y_min, x_max, y_max = bbxs[i]
        
        # Draw the bounding box on the image (color is blue, thickness is 2)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        # Write the frame with the bounding box to the video
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()

def generate_video(subset_name):
    bbx_txt_path = f"../test/tracking_results/jointnlt/swin_b_ep300_track/{subset_name}.txt"
    imgs_dir_path = f"../data/TNL2K_test/{subset_name}/imgs/"
    output_video_path = f"../test/tracking_results/jointnlt/swin_b_ep300_track/{subset_name}.mp4"
    
    # sort images
    imgs = [imgs_dir_path + img for img in os.listdir(imgs_dir_path) if img.endswith((".jpg", ".png"))]
    imgs.sort()
    
    # get bbxs
    bbxs = read_bbx(bbx_txt_path)

    assert(len(bbxs) == len(imgs))

    # Read the first image to get dimensions
    first_frame = cv2.imread(imgs[0])
    height, width, layers = first_frame.shape
    fps = 30
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    draw_bbxs(imgs, bbxs, video_writer)

if __name__ == "__main__":
    generate_video('test_001_BatMan_video_06')