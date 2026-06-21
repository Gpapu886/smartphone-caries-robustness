import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
from collections import Counter

# =========================
# 只改这里
# =========================
SRC_DATASET = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls'
OUT_DATASET = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls_augtrain'

# 每张训练图生成 6 张增强图，加上原图约 7 倍
AUG_PER_IMAGE = 6

RANDOM_SEED = 2026
# =========================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAMES = {
    0: 'no_caries',
    1: 'mild_caries',
    2: 'moderate_caries',
    3: 'severe_caries',
}


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def save_image(path, img):
    ext = Path(path).suffix
    success, encoded = cv2.imencode(ext, img)
    if success:
        encoded.tofile(str(path))


def copy_label(src_label, dst_label):
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)
    else:
        print(f'警告：找不到标签 {src_label}')


def adjust_label_for_hflip(src_label, dst_label):
    """
    YOLO格式：
    cls x_center y_center width height

    水平翻转后：
    x_center = 1 - x_center
    """
    if not os.path.exists(src_label):
        print(f'警告：找不到标签 {src_label}')
        return

    new_lines = []

    with open(src_label, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        if cls not in [0, 1, 2, 3]:
            print(f'警告：发现非4类标签 {cls} in {src_label}')
            continue

        x_new = 1.0 - x
        new_lines.append(f'{cls} {x_new:.6f} {y:.6f} {w:.6f} {h:.6f}')

    with open(dst_label, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')


def count_labels(label_dir):
    counter = Counter()

    label_files = list(Path(label_dir).glob('*.txt'))

    for label_path in label_files:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            counter[cls] += 1

    return counter


def aug_brightness(img):
    factor = random.uniform(0.65, 1.35)
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_contrast(img):
    alpha = random.uniform(0.65, 1.35)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    out = (img.astype(np.float32) - mean) * alpha + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 0] += random.uniform(-8, 8)
    hsv[:, :, 1] *= random.uniform(0.75, 1.25)
    hsv[:, :, 2] *= random.uniform(0.75, 1.25)

    hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def aug_gaussian_blur(img):
    k = random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_motion_blur(img):
    size = random.choice([5, 7, 9])
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)


def aug_jpeg(img):
    quality = random.choice([50, 60, 70, 80])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode('.jpg', img, encode_param)
    if not success:
        return img
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def aug_noise(img):
    sigma = random.uniform(5, 15)
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_hflip(img):
    return cv2.flip(img, 1)


AUG_FUNCS = [
    ('brightness', aug_brightness, False),
    ('contrast', aug_contrast, False),
    ('hsv', aug_hsv, False),
    ('gblur', aug_gaussian_blur, False),
    ('mblur', aug_motion_blur, False),
    ('jpeg', aug_jpeg, False),
    ('noise', aug_noise, False),
    ('hflip', aug_hflip, True),
]


def copy_val_test(split_name):
    src_img_dir = Path(SRC_DATASET) / split_name / 'images'
    src_label_dir = Path(SRC_DATASET) / split_name / 'labels'

    dst_img_dir = Path(OUT_DATASET) / split_name / 'images'
    dst_label_dir = Path(OUT_DATASET) / split_name / 'labels'

    mkdir(dst_img_dir)
    mkdir(dst_label_dir)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(src_img_dir.glob(ext)))

    print(f'复制 {split_name}: {len(image_files)} images')

    for img_path in image_files:
        shutil.copy(img_path, dst_img_dir / img_path.name)

        src_label = src_label_dir / f'{img_path.stem}.txt'
        dst_label = dst_label_dir / f'{img_path.stem}.txt'
        copy_label(src_label, dst_label)


def augment_train():
    src_img_dir = Path(SRC_DATASET) / 'train' / 'images'
    src_label_dir = Path(SRC_DATASET) / 'train' / 'labels'

    dst_img_dir = Path(OUT_DATASET) / 'train' / 'images'
    dst_label_dir = Path(OUT_DATASET) / 'train' / 'labels'

    mkdir(dst_img_dir)
    mkdir(dst_label_dir)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(src_img_dir.glob(ext)))

    image_files = sorted(image_files)

    print(f'训练原图数量：{len(image_files)}')

    original_count = 0
    aug_count = 0

    for img_path in image_files:
        img = read_image(img_path)

        if img is None:
            print(f'无法读取图片：{img_path}')
            continue

        src_label = src_label_dir / f'{img_path.stem}.txt'

        # 1. 复制原始 train 图片和标签
        shutil.copy(img_path, dst_img_dir / img_path.name)
        copy_label(src_label, dst_label_dir / f'{img_path.stem}.txt')
        original_count += 1

        # 2. 每张图生成 6 张增强图
        selected_augs = random.sample(AUG_FUNCS, AUG_PER_IMAGE)

        for idx, (aug_name, func, need_label_adjust) in enumerate(selected_augs):
            aug_img = func(img)

            new_stem = f'{img_path.stem}_aug_{aug_name}_{idx}'
            new_img_path = dst_img_dir / f'{new_stem}{img_path.suffix}'
            new_label_path = dst_label_dir / f'{new_stem}.txt'

            save_image(new_img_path, aug_img)

            if need_label_adjust:
                adjust_label_for_hflip(src_label, new_label_path)
            else:
                copy_label(src_label, new_label_path)

            aug_count += 1

    print(f'训练原图复制数量：{original_count}')
    print(f'训练增强图数量：{aug_count}')
    print(f'训练集总图片数量：{original_count + aug_count}')


def write_yaml():
    yaml_path = Path(OUT_DATASET) / 'caries_split_4cls_augtrain.yaml'

    content = f"""path: {str(Path(OUT_DATASET)).replace(chr(92), '/')}

train: train/images
val: val/images
test: test/images

nc: 4

names:
  0: no_caries
  1: mild_caries
  2: moderate_caries
  3: severe_caries
"""

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'已生成 yaml: {yaml_path}')


def print_dataset_stats():
    print('\n增强后类别实例统计：')

    for split in ['train', 'val', 'test']:
        label_dir = Path(OUT_DATASET) / split / 'labels'
        img_dir = Path(OUT_DATASET) / split / 'images'

        image_count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_count += len(list(img_dir.glob(ext)))

        counter = count_labels(label_dir)

        print(f'\n{split}: {image_count} images')
        for cls_id in range(4):
            print(f'  {cls_id} {CLASS_NAMES[cls_id]}: {counter[cls_id]}')


if __name__ == '__main__':
    if Path(OUT_DATASET).exists():
        print(f'删除旧文件夹：{OUT_DATASET}')
        shutil.rmtree(OUT_DATASET)

    augment_train()
    copy_val_test('val')
    copy_val_test('test')
    write_yaml()
    print_dataset_stats()

    print('\n完成：只增强 train，val/test 保持原始。')
    print(f'输出路径：{OUT_DATASET}')