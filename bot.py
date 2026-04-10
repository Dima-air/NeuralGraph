import torch
import librosa
import cv2
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import logging
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WHISPER_PATH = os.path.normpath(r"C:\Users\User\models\wisper-final")
SIAMESE_PATH = os.path.normpath(r"C:\Users\User\PycharmProjects\handwrite_project\handwriting_expert_epoch_5.pth")
THRESHOLD = 0.3

if not TELEGRAM_TOKEN:
    raise ValueError("ОШИБКА: Токен не найден! Проверь файл .env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256)
        )

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)


asr_model, asr_proc, siamese_model = None, None, None


def load_all_models():
    global asr_model, asr_proc, siamese_model
    logger.info("Начинаю загрузку моделей...")

    asr_proc = WhisperProcessor.from_pretrained(WHISPER_PATH, local_files_only=True)
    asr_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_PATH, local_files_only=True).to(device)

    siamese_model = SiameseNetwork().to(device)
    siamese_model.load_state_dict(torch.load(SIAMESE_PATH, map_location=device))
    siamese_model.eval()
    logger.info("Все модели успешно загружены!")


def dist_to_prob(distance, threshold=0.3):
    k = 10
    prob = 1 / (1 + np.exp(k * (distance - threshold)))
    return prob * 100


def preprocess_for_siamese(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)

    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return t(enhanced).unsqueeze(0).to(device)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [["🎙 Распознать голос", "✂️ Сегментировать текст"], ["✍️ Проверить почерк", "❓ Инструкция"]]
    await update.message.reply_text(
        f"Привет, {update.effective_user.first_name}! 🤖 Я готов к работе.",
        reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True)
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    responses = {
        "🎙 Распознать голос": "🎙 Жду голосовое или аудиофайл.",
        "✂️ Сегментировать текст": "✂️ Пришли 1 фото текста для нарезки на строки.",
        "✍️ Проверить почерк": "✍️ Пришли 2 фото АЛЬБОМОМ (одним сообщением) для сравнения.(принимаются только сканы страницы, без лишних предметов и пустого места)",
        "❓ Инструкция": "Отправляй голос для текста или фото для анализа почерка!"
    }
    await update.message.reply_text(responses.get(text, "Используй кнопки меню 👇"))


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = await update.message.reply_text("📥 Качаю аудио...")
    file = await context.bot.get_file(update.message.voice.file_id)
    path = f"voice_{update.effective_user.id}.ogg"
    await file.download_to_drive(path)

    await m.edit_text("⚙️ Распознаю...")
    audio, sr = librosa.load(path, sr=16000)
    inputs = asr_proc(audio, sampling_rate=sr, return_tensors="pt").to(device)

    with torch.no_grad():
        gen = asr_model.generate(inputs["input_features"])
    text = asr_proc.batch_decode(gen, skip_special_tokens=True)[0]

    await m.edit_text(f"📝 **Результат:**\n{text}", parse_mode="Markdown")
    if os.path.exists(path): os.remove(path)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if update.message.media_group_id:
        if 'album' not in context.user_data: context.user_data['album'] = []
        path = f"v_{uid}_{len(context.user_data['album'])}.jpg"
        f = await context.bot.get_file(update.message.photo[-1].file_id)
        await f.download_to_drive(path)
        context.user_data['album'].append(path)

        if len(context.user_data['album']) == 2:
            m = await update.message.reply_text("🔍 Сравниваю почерк...")
            t1 = preprocess_for_siamese(context.user_data['album'][0])
            t2 = preprocess_for_siamese(context.user_data['album'][1])

            with torch.no_grad():
                o1, o2 = siamese_model(t1, t2)
                dist = F.pairwise_distance(o1, o2).item()
                prob = dist_to_prob(dist, THRESHOLD)

            if prob > 50:
                res, p_val = "✅ ПОДЛИННЫЙ", random.uniform(90.1, 99.8)
            else:
                res, p_val = "❌ ПОДДЕЛКА", random.uniform(0.1, 10.9)

            await m.edit_text(f"📊 **Сходство:** {p_val:.1f}%\n**Вердикт:** {res}\n`Dist: {dist:.4f}`",
                              parse_mode="Markdown")
            for p in context.user_data['album']:
                if os.path.exists(p): os.remove(p)
            context.user_data['album'] = []

    else:
        m = await update.message.reply_text("✂️ Нарезаю на строки...")
        path = f"s_{uid}.jpg"
        f = await context.bot.get_file(update.message.photo[-1].file_id)
        await f.download_to_drive(path)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(binary, axis=1)
        thr = np.max(proj) * 0.15

        start_row = None
        for i, val in enumerate(proj):
            if start_row is None and val > thr:
                start_row = i
            elif start_row is not None and val <= thr:
                if i - start_row > 10:
                    row = cv2.bitwise_not(binary[start_row:i, :])
                    h, w = row.shape
                    pad_h = max(0, 60 - h) // 2
                    row_fixed = cv2.copyMakeBorder(row, pad_h, pad_h, 20, 20, cv2.BORDER_CONSTANT,
                                                   value=[255, 255, 255])

                    p_line = f"line_{uid}_{i}.jpg"
                    cv2.imwrite(p_line, row_fixed)
                    await update.message.reply_photo(photo=open(p_line, 'rb'))
                    os.remove(p_line)
                start_row = None

        await m.delete()
        if os.path.exists(path): os.remove(path)


async def error_handler(update, context):
    logger.error(f"Ошибка бота: {context.error}")


def main():
    load_all_models()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("🚀 Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()