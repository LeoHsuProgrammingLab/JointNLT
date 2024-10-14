from PIL import Image
import os
from tqdm import tqdm

class DatasetChecker():
    def __init__(self, img_dir):
        self.img_dir = img_dir
    
    def check(self):
        img_names = sorted([img for img in os.listdir(self.img_dir) if img.endswith((".jpg", ".png"))])
        for img_name in img_names:
            try:
                img = Image.open(self.img_dir + img_name)
                img.verify()
            except Exception as e:
                print(f"Error in {img_name}: {e}")

if __name__ == "__main__":
    dir_name = "/scratch/user/agenuinedream/JointNLT/data/TNL2K_test"
    img_dirs = [f"{dir_name}/{subset_name}/imgs/" for subset_name in os.listdir(dir_name)]
    
    for img_dir in tqdm(img_dirs):
        checker = DatasetChecker(img_dir)
        checker.check()