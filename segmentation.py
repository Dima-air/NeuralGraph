import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def segment_lines(img_path, output_dir="output_lines", debug=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Loading: {img_path} (name: {img_name})")

    img_array = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Couldn't load: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Binarization")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    print("Projection analysis")
    projection = np.sum(binary, axis=1)

    max_proj = np.max(projection)
    empty_threshold = max_proj * 0.15

    line_starts = []
    line_ends = []
    in_text = False

    for i, val in enumerate(projection):
        if not in_text and val > empty_threshold:
            line_starts.append(i)
            in_text = True
        elif in_text and val <= empty_threshold:
            if i - line_starts[-1] > 10:
                line_ends.append(i)
            else:
                line_starts.pop()
            in_text = False

    if in_text and len(line_starts) > 0:
        if len(projection) - line_starts[-1] > 10:
            line_ends.append(len(projection) - 1)
        else:
            line_starts.pop()

    print(f"Strings found: {len(line_starts)}")

    print("Strings slice")
    saved = 0
    for i, (start, end) in enumerate(zip(line_starts, line_ends)):
        margin = 3
        line_img = binary[max(0, start - margin):min(binary.shape[0], end + margin), :]
        line_img = cv2.bitwise_not(line_img)

        file_name = f"{img_name}_line_{saved}.jpg"
        cv2.imwrite(os.path.join(output_dir, file_name), line_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if saved % 10 == 0:
            print(f"Saved {saved}/{len(line_starts)}")

    print(f"READY. Saved {saved} strings in {output_dir}/")
    print(f"Files: {img_name}_line_0.jpg, {img_name}_line_1.jpg, ...")

    if debug:
        print("Graph generation")
        plt.figure(figsize=(15, 6))
        plt.plot(projection, '-b', linewidth=1, label='Projection')
        plt.axhline(y=empty_threshold, color='r', linestyle='--', label=f'Threshold={empty_threshold:.0f}')
        plt.title(f"Horizontal projection (strings found: {len(line_starts)})")
        plt.xlabel("Y coordinate")
        plt.ylabel("Sum of pixels")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{img_name}_projection.png", dpi=100, bbox_inches='tight')
        print(f"Graph: {img_name}_projection.png")
        plt.close()

    return saved


if __name__ == "__main__":
    segment_lines(r"C:\Users\User\datasets\HWR200_Direct\hwr200_100_119\hw_dataset_cropped\118\reuse4\Сканы\3.jpg", debug=False)