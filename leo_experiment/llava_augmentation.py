import torch
import requests
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

class LLaVA_Engine():
    def __init__(self, cache_dir, model_name="llava-hf/llava-1.5-7b-hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            cache_dir=cache_dir
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
    
    def conversation_template(self, text_content, role="user"):
        return [
            {
                "role": role, 
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": text_content},
                ],
            },
        ]
    
    def prompt_template(self, desc_template, desc_refined="", template_id=0):
#                 This is the updated description of tracking target object in the previous frame: {desc_refined}. \n \
        prompt_list = [
            f"""
                This is the original description of tracked target object in the first frame: {desc_template}.
                Please do the following:
                1. Describe this object in current frame more concisely, updated to fit the current situation in one sentence.
                2. Output the bounding box coordinates of this object in [x1, y1, x2, y2] format. If the target object doesn't exist in current frame, please output coordinate [0, 0, 0, 0].
                Please format the output as a single line, with the description and bounding box separated by '#': updated_description #bounding_box
            """
        ]
        
        return prompt_list[template_id]
    
    def process_input(self, imgs, prompts):   
        template_prompts = [self.processor.apply_chat_template(self.conversation_template(prompt), add_generation_prompt=True) for prompt in prompts]
        inputs = self.processor(images=imgs, text=template_prompts, padding=True, return_tensors="pt").to(self.model.device)
        return inputs

    def batch_output(self, imgs, prompts, max_new_tokens=30):
        inputs = self.process_input(imgs, prompts)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)   
        return outputs

def main():
    output_dir = '/scratch/user/agenuinedream/JointNLT/test/tracking_results/jointnlt/swin_b_ep300_track/llava_text'
    data_dir = '/scratch/user/agenuinedream/JointNLT/data/TNL2K_test'
    data_list = sorted(os.listdir(data_dir))
    llava = LLaVA_Engine(cache_dir="/scratch/user/agenuinedream/.cache/huggingface")

    for i, data in tqdm(enumerate(data_list), total=len(data_list)):
        img_dir = os.path.join(data_dir, data, 'imgs')
        text_file = os.path.join(data_dir, data, 'language.txt')
        with open(text_file, 'r') as f:
            desc = f.read().strip()
           
        prompt = llava.prompt_template(desc)
        img_list = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        imgs = [Image.open(os.path.join(img_dir, img_name)) for img_name in img_list]
        
        batch_size = 20

        # handle output file
        output_file = os.path.join(output_dir, f'{data}_llava.txt')
        if os.path.exists(output_file):
            os.remove(output_file)

        for i in range(0, len(imgs), batch_size):
            batch_imgs = imgs[i:i + batch_size]
            prompts = [prompt] * len(batch_imgs)
            outputs = llava.batch_output(batch_imgs, prompts, 70)
            for output in outputs:
                with open(output_file, 'a') as f:
                    f.write(output.split('ASSISTANT:')[-1].strip() + '\n')

# def original_decode():
#     cache_dir = "/scratch/user/agenuinedream/.cache/huggingface"
#     model = LlavaForConditionalGeneration.from_pretrained(
#         "llava-hf/llava-1.5-7b-hf", 
#         torch_dtype=torch.float16, 
#         device_map="auto",
#         cache_dir=cache_dir
#     )
#     processor = AutoProcessor.from_pretrained(
#         "llava-hf/llava-1.5-7b-hf",
#         cache_dir=cache_dir
#     )

#     url = "https://www.ilankelman.org/stopsigns/australia.jpg"
#     image_stop = Image.open(requests.get(url, stream=True).raw)

#     url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     image_cats = Image.open(requests.get(url, stream=True).raw)

#     conservation_1 = [
#         {
#             "role": "user", 
#             "content": [
#                 {"type": "image"}, 
#                 {"type": "text", "text": "Describe this image."},
#             ],
#         },
#     ]

#     prompt_1 = processor.apply_chat_template(conservation_1, add_generation_prompt=True)
#     prompts = [prompt_1, prompt_1]
#     inputs = processor(images=[image_stop, image_cats], text=prompts, padding=True, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_new_tokens=60)
#         outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)   

#     for i, output in enumerate(outputs):
#         print(f"Output {i}: {output.split('ASSISTANT:')[-1].strip()}")

if __name__ == "__main__": 
    main()
