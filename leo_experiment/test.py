import torch

if __name__ == "__main__":
    # check if gpu available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)   