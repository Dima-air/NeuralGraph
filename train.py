import os.path

import torch
from torch.utils.data import DataLoader
from dataset import SiameseHandwritingDataset
from model import SiameseNetwork, ContrastiveLoss
import torchvision.transforms as transforms

def train():
    global running_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем {device}")

    #трансформы
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    try:
        dataset = SiameseHandwritingDataset(data_dir="C:/Users/User/Desktop/dataset", transform=train_transform)
    except ValueError as e:
        print(f"Ошибка датасета: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = SiameseNetwork().to(device)

    #загрузка последней сохрененой неросети
    latest_epoch = 0
    for epoch_num in range(50, 0, -1):
        model_path = f"siamese_epoch_{epoch_num}.pth"
        if os.path.exists(model_path):
            print(f"Найдена модель после {epoch_num} эпох, идет ее загрузка")
            model.load_state_dict(torch.load(model_path, map_location=device))
            latest_epoch = epoch_num
            break
    start_epoch = latest_epoch

    criterion = ContrastiveLoss(margin=1.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 50
    print(f"Начало обучение на {epochs} эпох. Начинаем с {start_epoch+1}")

    for epoch in range(start_epoch, 50):
        model.train()
        running_loss = 0.0

        for i, (img1, img2, label) in enumerate(dataloader):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()

            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label.view(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Эпоха [{epoch}/{epochs}], Батч [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Эпоха {epoch + 1}/{epochs} завершена. Средняя ошибка {avg_loss}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"siamese_epoch_{epoch+1}.pth")
            print(f"Эпоха {epoch} сохранена. Средний Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()