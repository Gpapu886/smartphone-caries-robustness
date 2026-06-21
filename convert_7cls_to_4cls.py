import os
import shutil
from pathlib import Path
from collections import Counter

# =========================
# 只改这里
# =========================
SRC_DATASET = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split'
OUT_DATASET = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls'
# =========================

CLASS_MAP = {
    0: 0,  # no_caries
    1: 1,  # mild_caries
    2: 1,  # mild_caries
    3: 2,  # moderate_caries
    4: 2,  # moderate_caries
    5: 3,  # severe_caries
    6: 3,  # severe_caries
}

NEW_NAMES = {
    0: 'no_caries',
    1: 'mild_caries',
    2: 'moderate_caries',
    3: 'severe_caries',
}


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def convert_label(src_label, dst_label):
    """
    把一个 YOLO txt 标签从7类转成4类。
    YOLO格式：
    cls x_center y_center width height
    """
    new_lines = []
    class_counter = Counter()

    if not os.path.exists(src_label):
        return class_counter

    with open(src_label, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        old_cls = int(float(parts[0]))

        if old_cls not in CLASS_MAP:
            print(f'警告：发现未知类别 {old_cls} in {src_label}')
            continue

        new_cls = CLASS_MAP[old_cls]
        parts[0] = str(new_cls)

        new_lines.append(' '.join(parts))
        class_counter[new_cls] += 1

    with open(dst_label, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')

    return class_counter


def process_split(split_name):
    src_img_dir = Path(SRC_DATASET) / split_name / 'images'
    src_label_dir = Path(SRC_DATASET) / split_name / 'labels'

    dst_img_dir = Path(OUT_DATASET) / split_name / 'images'
    dst_label_dir = Path(OUT_DATASET) / split_name / 'labels'

    make_dir(dst_img_dir)
    make_dir(dst_label_dir)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(src_img_dir.glob(ext)))

    print(f'\n处理 {split_name}: {len(image_files)} images')

    split_counter = Counter()

    for img_path in image_files:
        # 复制图片
        shutil.copy(img_path, dst_img_dir / img_path.name)

        # 转换标签
        src_label = src_label_dir / f'{img_path.stem}.txt'
        dst_label = dst_label_dir / f'{img_path.stem}.txt'

        counter = convert_label(src_label, dst_label)
        split_counter.update(counter)

    print(f'{split_name} 类别实例统计：')
    for cls_id in range(4):
        print(f'  {cls_id} {NEW_NAMES[cls_id]}: {split_counter[cls_id]}')

    return split_counter


def write_yaml():
    yaml_path = Path(OUT_DATASET) / 'caries_split_4cls.yaml'

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

    print(f'\n已生成 yaml: {yaml_path}')


if __name__ == '__main__':
    if Path(OUT_DATASET).exists():
        print(f'删除旧文件夹：{OUT_DATASET}')
        shutil.rmtree(OUT_DATASET)

    total_counter = Counter()

    for split in ['train', 'val', 'test']:
        counter = process_split(split)
        total_counter.update(counter)

    print('\n全部数据类别实例统计：')
    for cls_id in range(4):
        print(f'  {cls_id} {NEW_NAMES[cls_id]}: {total_counter[cls_id]}')

    write_yaml()

    print('\n7类转4类完成。')
    print(f'输出路径：{OUT_DATASET}')