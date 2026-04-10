import torch
import torch.nn.functional as F
from model import SiameseNetwork
from dataset import SiameseHandwritingDataset
import torchvision.transforms as transform
import cv2

def verify(img1_path, img2_path, model_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ds_helper = SiameseHandwritingDataset("C:/Users/User/Desktop/dataset")
    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Normalize((0.5,), (0.5,))
    ])

    img1_cv = cv2.imread(img1_path)
    img2_cv = cv2.imread(img2_path)

    img1_pil = ds_helper.opencv_preprocess(img1_cv)
    img2_pil = ds_helper.opencv_preprocess(img2_cv)

    img1 = transforms(img1_pil).unsqueeze(0).to(device)
    img2 = transforms(img2_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1, emb2 = model(img1, img2)
        dist = F.pairwise_distance(emb1, emb2).item()

    is_same = dist < threshold
    print("Сравнение почерка")
    print(f"Дистанция: {dist:.4f}")
    print(f"Порог: {threshold}")
    print(f"Результат {'SAME' if is_same else 'NOT SAME'}")

if __name__ == "__main__":
    verify("C:/Users/User/Desktop/dataset/11_s.jpg",
           "C:/Users/User/Desktop/dataset/10_s.jpg",
           "siamese_epoch_15.pth")