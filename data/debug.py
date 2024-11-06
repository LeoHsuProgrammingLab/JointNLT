import os
from PIL import Image
from tqdm import tqdm
import shutil

def find_error_img(dir_path):
    img_names = sorted([dir_path + img for img in os.listdir(dir_path) if img.endswith((".jpg", ".png"))])
    for img_name in tqdm(img_names):
        try:
            image = Image.open(img_name)
            image.load()
        except OSError as e:
            print(f"Cannot open image: {e}")
            print(f"Image name: {img_name}")

def mv_data(src_dir, tgt_dir):
    for subdir in os.listdir(src_dir):
        subdir_path = os.path.join(src_dir, subdir)
        
        # Check if the item is a directory
        if os.path.isdir(subdir_path):
            file_path = os.path.join(subdir_path, 'llava_output.txt')
            
            # Check if llava_output.txt exists in the subdirectory
            if os.path.exists(file_path):
                # Define the new file name as <subdir>_llava_output.txt
                new_file_name = f"{subdir}_llava.txt"
                new_file_path = os.path.join(tgt_dir, new_file_name)
                
                # Move and rename the file
                shutil.move(file_path, new_file_path)
                print(f"Moved {file_path} to {new_file_path}")

def main():
    src_dir = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test'
    tgt_dir = '/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/llava_text'
    mv_data(src_dir, tgt_dir)

if __name__ == "__main__":
    main()