import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SiameseNetwork
from dataset import SiameseHandwritingDataset
from PIL import Image
import torchvision.transforms as transforms

def get_attention_map(img_path, model_path):
    device = torch.device("cpu")

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ds_helper = SiameseHandwritingDataset("C:/Users/User/Desktop/dataset")

    img_cv = cv2.imread(img_path)
    img_pil = ds_helper.opencv_preprocess(img_cv)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    embedding = model.get_embedding(input_tensor)
    score = embedding.norm()
    score.backward()

    saliency =  input_tensor.grad.abs().squeeze()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Что видит нейронка (Binary)")
    plt.imshow(img_pil, cmap='gray')

    plt.subplot(1, 2 , 2)
    plt.title("Карта внимания")
    plt.imshow(saliency, cmap='hot')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    get_attention_map("C:/Users/User/Desktop/dataset/2_b.jpg", "siamese_epoch_15.pth")