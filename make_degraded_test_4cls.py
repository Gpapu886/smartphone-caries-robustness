import os
import cv2
import yaml
import shutil
import random
import numpy as np
from pathlib import Path

# =========================
# 只改这里
# =========================
SRC_DATASET = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls_augtrain'
OUT_ROOT = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_4cls_degraded_test'
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


def low_light(img, level):
    if level == 'mild':
        factor = 0.75
    elif level == 'medium':
        factor = 0.55
    else:
        factor = 0.35

    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def over_exposure(img, level):
    if level == 'mild':
        factor = 1.25
    elif level == 'medium':
        factor = 1.50
    else:
        factor = 1.80

    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(img, level):
    if level == 'mild':
        k = 3
    elif level == 'medium':
        k = 5
    else:
        k = 9

    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, level):
    if level == 'mild':
        size = 5
    elif level == 'medium':
        size = 9
    else:
        size = 13

    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size)
    kernel = kernel / size

    return cv2.filter2D(img, -1, kernel)


def jpeg_compression(img, level):
    if level == 'mild':
        quality = 80
    elif level == 'medium':
        quality = 60
    else:
        quality = 40

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode('.jpg', img, encode_param)

    if not success:
        return img

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded


def random_occlusion(img, level):
    h, w = img.shape[:2]

    if level == 'mild':
        ratio = 0.10
    elif level == 'medium':
        ratio = 0.20
    else:
        ratio = 0.30

    occ_w = int(w * ratio)
    occ_h = int(h * ratio)

    x1 = random.randint(0, max(1, w - occ_w))
    y1 = random.randint(0, max(1, h - occ_h))

    out = img.copy()
    out[y1:y1 + occ_h, x1:x1 + occ_w] = 0

    return out


def contrast_shift(img, level):
    if level == 'mild':
        alpha = 0.80
    elif level == 'medium':
        alpha = 0.60
    else:
        alpha = 0.40

    mean = np.mean(img, axis=(0, 1), keepdims=True)
    out = (img.astype(np.float32) - mean) * alpha + mean

    return np.clip(out, 0, 255).astype(np.uint8)


def color_shift(img, level):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if level == 'mild':
        shift = 5
    elif level == 'medium':
        shift = 10
    else:
        shift = 15

    hsv[:, :, 0] += shift
    hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


DEGRADATIONS = {
    'low_light': low_light,
    'over_exposure': over_exposure,
    'gaussian_blur': gaussian_blur,
    'motion_blur': motion_blur,
    'jpeg_compression': jpeg_compression,
    'occlusion': random_occlusion,
    'contrast_shift': contrast_shift,
    'color_shift': color_shift,
}


def write_yaml(out_root):
    yaml_path = Path(out_root) / 'data.yaml'

    data_yaml = {
        'path': str(Path(out_root)).replace('\\', '/'),
        'train': 'test/images',
        'val': 'test/images',
        'test': 'test/images',
        'nc': 4,
        'names': CLASS_NAMES
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    return yaml_path


def make_one_degraded_dataset(deg_name, level, func):
    src_img_dir = Path(SRC_DATASET) / 'test' / 'images'
    src_label_dir = Path(SRC_DATASET) / 'test' / 'labels'

    out_name = f'{deg_name}_{level}'
    out_root = Path(OUT_ROOT) / out_name

    out_img_dir = out_root / 'test' / 'images'
    out_label_dir = out_root / 'test' / 'labels'

    mkdir(out_img_dir)
    mkdir(out_label_dir)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(src_img_dir.glob(ext)))

    print(f'正在生成 {out_name}: {len(image_files)} images')

    for img_path in image_files:
        img = read_image(img_path)

        if img is None:
            print(f'无法读取图片：{img_path}')
            continue

        out_img = func(img, level)

        save_image(out_img_dir / img_path.name, out_img)

        src_label = src_label_dir / f'{img_path.stem}.txt'
        dst_label = out_label_dir / f'{img_path.stem}.txt'

        if src_label.exists():
            shutil.copy(src_label, dst_label)
        else:
            print(f'警告：找不到标签 {src_label}')

    yaml_path = write_yaml(out_root)
    print(f'已生成 yaml: {yaml_path}')


if __name__ == '__main__':
    if Path(OUT_ROOT).exists():
        print(f'删除旧文件夹：{OUT_ROOT}')
        shutil.rmtree(OUT_ROOT)

    for deg_name, func in DEGRADATIONS.items():
        for level in ['mild', 'medium', 'severe']:
            make_one_degraded_dataset(deg_name, level, func)

    print('\n全部 degraded test 生成完成。')
    print(f'输出路径：{OUT_ROOT}')