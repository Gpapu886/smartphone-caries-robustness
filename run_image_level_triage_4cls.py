import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from ultralytics import YOLO

# =========================
# 只改这里
# =========================
WEIGHT_PATH = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/train/V8n_4cls_augtrain/weights/best.pt'

CLEAN_IMG_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls_augtrain/test/images'
CLEAN_LABEL_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls_augtrain/test/labels'

DEGRADED_ROOT = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_4cls_degraded_test'

OUT_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/triage_summary_4cls'

CONF_THRES = 0.25
# =========================


TRIAGE_TASKS = {
    # true/pred image-level max class >= threshold
    'any_caries_cls_ge_1': 1,
    'moderate_or_severe_cls_ge_2': 2,
    'severe_caries_cls_ge_3': 3,
}


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def get_image_files(img_dir):
    img_dir = Path(img_dir)
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files.extend(list(img_dir.glob(ext)))
    return sorted(files)


def read_max_gt_class(label_path):
    """
    读取真实标签，返回一张图中最高严重程度类别。
    如果没有标签，返回 -1。
    """
    if not os.path.exists(label_path):
        return -1

    max_cls = -1

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        max_cls = max(max_cls, cls)

    return max_cls


def read_max_pred_class(pred_label_path):
    """
    读取预测标签，返回一张图中预测到的最高严重程度类别。
    YOLO预测txt格式：
    cls x y w h conf
    如果没有预测框，返回 -1。
    """
    if not os.path.exists(pred_label_path):
        return -1

    max_cls = -1

    with open(pred_label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        max_cls = max(max_cls, cls)

    return max_cls


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else np.nan

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'NPV': npv,
        'F1': f1,
        'accuracy': accuracy,
    }


def evaluate_condition(condition, img_dir, gt_label_dir, pred_label_dir):
    image_files = get_image_files(img_dir)

    rows = []

    gt_max_list = []
    pred_max_list = []

    for img_path in image_files:
        stem = img_path.stem

        gt_label = Path(gt_label_dir) / f'{stem}.txt'
        pred_label = Path(pred_label_dir) / f'{stem}.txt'

        gt_max = read_max_gt_class(gt_label)
        pred_max = read_max_pred_class(pred_label)

        gt_max_list.append(gt_max)
        pred_max_list.append(pred_max)

    for task_name, threshold in TRIAGE_TASKS.items():
        y_true = [1 if x >= threshold else 0 for x in gt_max_list]
        y_pred = [1 if x >= threshold else 0 for x in pred_max_list]

        metrics = compute_metrics(y_true, y_pred)

        row = {
            'condition': condition,
            'task': task_name,
            'threshold_class': threshold,
            'num_images': len(image_files),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def run_predict(model, condition, img_dir):
    pred_project = Path(OUT_DIR) / 'prediction_labels'

    model.predict(
        source=str(img_dir),
        imgsz=640,
        conf=CONF_THRES,
        device='0',
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(pred_project),
        name=condition,
        exist_ok=True,
        verbose=False,
    )

    pred_label_dir = pred_project / condition / 'labels'

    return pred_label_dir


def get_conditions():
    conditions = []

    # clean
    conditions.append({
        'condition': 'clean_test',
        'img_dir': CLEAN_IMG_DIR,
        'label_dir': CLEAN_LABEL_DIR,
    })

    # degraded
    degraded_root = Path(DEGRADED_ROOT)
    degraded_sets = sorted([p for p in degraded_root.iterdir() if p.is_dir()])

    for ds in degraded_sets:
        conditions.append({
            'condition': ds.name,
            'img_dir': str(ds / 'test' / 'images'),
            'label_dir': str(ds / 'test' / 'labels'),
        })

    return conditions


if __name__ == '__main__':
    print('检查权重是否存在：', os.path.exists(WEIGHT_PATH))
    print('检查 clean images 是否存在：', os.path.exists(CLEAN_IMG_DIR))
    print('检查 clean labels 是否存在：', os.path.exists(CLEAN_LABEL_DIR))
    print('检查 degraded root 是否存在：', os.path.exists(DEGRADED_ROOT))

    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(WEIGHT_PATH)

    mkdir(OUT_DIR)

    # 清理旧预测标签，避免混入旧结果
    pred_root = Path(OUT_DIR) / 'prediction_labels'
    if pred_root.exists():
        shutil.rmtree(pred_root)

    model = YOLO(WEIGHT_PATH)

    all_rows = []

    conditions = get_conditions()

    for item in conditions:
        condition = item['condition']
        img_dir = item['img_dir']
        gt_label_dir = item['label_dir']

        print(f'\n正在预测并评估：{condition}')

        pred_label_dir = run_predict(model, condition, img_dir)

        rows = evaluate_condition(
            condition=condition,
            img_dir=img_dir,
            gt_label_dir=gt_label_dir,
            pred_label_dir=pred_label_dir,
        )

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    save_all = Path(OUT_DIR) / 'image_level_triage_all_conditions.csv'
    df.to_csv(save_all, index=False, encoding='utf-8-sig')

    # 主文精选条件
    main_conditions = [
        'clean_test',
        'color_shift_severe',
        'contrast_shift_severe',
        'low_light_severe',
        'over_exposure_severe',
        'occlusion_severe',
    ]

    df_main = df[df['condition'].isin(main_conditions)].copy()
    save_main = Path(OUT_DIR) / 'image_level_triage_main_conditions.csv'
    df_main.to_csv(save_main, index=False, encoding='utf-8-sig')

    print('\n全部完成。')
    print('全部条件结果：', save_all)
    print('主文精选结果：', save_main)

    print('\n主文精选结果预览：')
    print(df_main)