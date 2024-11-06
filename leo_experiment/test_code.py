import torch
import clip

if __name__ == "__main__":
    # check if gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)   
    # Load the model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Prepare an example input
    text = clip.tokenize(["a photo of a cat"]).to(device)

    # Run the model
    with torch.no_grad():
        text_features = model.encode_text(text)

    print("Text features shape:", text_features.shape)