import os
from PIL import Image
from tqdm import tqdm

def find_error_img(dir_path):
    img_names = sorted([dir_path + img for img in os.listdir(dir_path) if img.endswith((".jpg", ".png"))])
    for img_name in tqdm(img_names):
        try:
            image = Image.open(img_name)
            image.load()
        except OSError as e:
            print(f"Cannot open image: {e}")
            print(f"Image name: {img_name}")

if __name__ == "__main__":
    find_error_img("/scratch/user/agenuinedream/JointNLT/data/TNL2K_test/test_015_Sord_video_Q01_done/imgs/")