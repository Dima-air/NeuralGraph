import os
import cv2
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class SiameseHandwritingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))])

        self.my_hand = [os.path.join(data_dir, f) for f in all_files if '_s' in f]
        self.other_hand = [os.path.join(data_dir,f) for f in all_files if '_b' in f]

        if len(self.my_hand) < 2:
            raise ValueError("Минимум 2 изображения своего почерка")
        if len(self.other_hand) == 0:
            raise ValueError("Хотя бы 1 изображение чужого почерка")
        print(f"мой почерк - {len(self.my_hand)}, чужой - {len(self.other_hand)}")


    def opencv_preprocess(self, cv_img):
        if cv_img is None:
            return None

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        resized = cv2.resize(binary, (128, 128))
        return Image.fromarray(resized)

    def __len__(self):
        return len(self.my_hand) * 10

    def __getitem__(self, item, retry_count=0):
        label = random.randint(0, 1)

        if label == 1:
            if len(self.my_hand) >= 2:
                img1_path, img2_path = random.sample(self.my_hand, 2)
            else:
                img1_path = self.my_hand[0]
                img2_path = self.my_hand[0]
        else:
            img1_path = random.choice(self.my_hand)
            img2_path = random.choice(self.other_hand)

        img1_cv = cv2.imread(img1_path)
        img2_cv = cv2.imread(img2_path)

        if img1_cv is None or img2_cv is None:
            if retry_count > 5:
                raise RuntimeError(f'Не удалось загрузить после {retry_count} попыток')
            return self.__getitem__(item, retry_count + 1)

        img1 = self.opencv_preprocess(img1_cv)
        img2 = self.opencv_preprocess(img2_cv)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)