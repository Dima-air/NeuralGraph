import pytesseract
from PIL import Image
import os
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def recognize_line(img_path):
    try:
        img = Image.open(img_path)
        custom_config = r'--oem 3 --psm 7 -l rus'
        text = pytesseract.image_to_string(img, config=custom_config)
        return text.strip()
    except Exception as e:
        return f"Error: {e}"


input_dir = "output_lines"
output_md = "recognized_tesseract.md"

results = {}

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith('.jpg'):
        filepath = os.path.join(input_dir, filename)

        match = re.match(r'(.+)_line_(\d+)\.jpg', filename)
        img_name = match.group(1) if match else filename
        line_num = int(match.group(2)) if match else 0

        text = recognize_line(filepath)

        if text:
            if img_name not in results: results[img_name] = {}
            results[img_name][line_num] = text
            print(f"{filename}: {text[:50]}...")
        else:
            print(f"{filename}: пусто")

with open(output_md, 'w', encoding='utf-8') as f:
    f.write("Recognized text\n\n")
    for img_name in sorted(results.keys()):
        f.write(f"## {img_name}\n\n")
        lines = results[img_name]
        for line_num in sorted(lines.keys()):
            f.write(f"{lines[line_num]}\n")
        f.write("\n---\n\n")

print(f"\nReady, result: {output_md}")