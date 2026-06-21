import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
from ultralytics import YOLO

# =========================
# 只改这里
# =========================
WEIGHT_PATH = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/train/V8n_4cls_augtrain/weights/best.pt'
CLEAN_YAML = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls_augtrain/caries_split_4cls_augtrain.yaml'
DEGRADED_ROOT = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_4cls_degraded_test'
OUT_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/robust_summary_4cls'
# =========================

CLASS_NAMES = {
    0: 'no_caries',
    1: 'mild_caries',
    2: 'moderate_caries',
    3: 'severe_caries',
}


def arr(x):
    return np.asarray(x, dtype=float).reshape(-1)


def safe_get(a, i):
    if len(a) > i:
        return float(a[i])
    return np.nan


def collect_one_result(condition, metrics):
    rows = []

    # 总体结果
    rows.append({
        'condition': condition,
        'class_id': 'all',
        'class_name': 'all',
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
    })

    # 每类结果
    p_cls = arr(metrics.box.p)
    r_cls = arr(metrics.box.r)
    ap50_cls = arr(metrics.box.ap50)
    map_cls = arr(metrics.box.maps)

    for cls_id in range(4):
        rows.append({
            'condition': condition,
            'class_id': cls_id,
            'class_name': CLASS_NAMES[cls_id],
            'precision': safe_get(p_cls, cls_id),
            'recall': safe_get(r_cls, cls_id),
            'mAP50': safe_get(ap50_cls, cls_id),
            'mAP50_95': safe_get(map_cls, cls_id),
        })

    return rows


def sort_key(path):
    name = path.name

    level_order = {
        'mild': 1,
        'medium': 2,
        'severe': 3,
    }

    parts = name.split('_')
    level = parts[-1]
    deg = '_'.join(parts[:-1])

    return deg, level_order.get(level, 99)


if __name__ == '__main__':
    print('检查权重是否存在：', os.path.exists(WEIGHT_PATH))
    print('检查 clean yaml 是否存在：', os.path.exists(CLEAN_YAML))
    print('检查 degraded root 是否存在：', os.path.exists(DEGRADED_ROOT))

    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(WEIGHT_PATH)

    if not os.path.exists(CLEAN_YAML):
        raise FileNotFoundError(CLEAN_YAML)

    if not os.path.exists(DEGRADED_ROOT):
        raise FileNotFoundError(DEGRADED_ROOT)

    os.makedirs(OUT_DIR, exist_ok=True)

    model = YOLO(WEIGHT_PATH)

    all_rows = []

    # clean test
    print('\n正在验证 clean_test...')
    metrics = model.val(
        data=CLEAN_YAML,
        split='test',
        imgsz=640,
        batch=16,
        device='0',
        workers=0,
        plots=False,
        project='runs/robust_val_4cls_perclass',
        name='clean_test'
    )
    all_rows.extend(collect_one_result('clean_test', metrics))

    # degraded test
    degraded_sets = sorted(
        [p for p in Path(DEGRADED_ROOT).iterdir() if p.is_dir()],
        key=sort_key
    )

    for ds in degraded_sets:
        condition = ds.name
        yaml_path = ds / 'data.yaml'

        if not yaml_path.exists():
            print(f'跳过 {condition}: 找不到 data.yaml')
            continue

        print(f'\n正在验证 {condition}...')

        metrics = model.val(
            data=str(yaml_path),
            split='test',
            imgsz=640,
            batch=16,
            device='0',
            workers=0,
            plots=False,
            project='runs/robust_val_4cls_perclass',
            name=condition
        )

        all_rows.extend(collect_one_result(condition, metrics))

    df = pd.DataFrame(all_rows)

    # 计算相对 clean 的下降
    clean_df = df[df['condition'] == 'clean_test'].copy()
    clean_lookup = {}

    for _, row in clean_df.iterrows():
        key = str(row['class_id'])
        clean_lookup[key] = {
            'precision': row['precision'],
            'recall': row['recall'],
            'mAP50': row['mAP50'],
            'mAP50_95': row['mAP50_95'],
        }

    drops = []

    for _, row in df.iterrows():
        key = str(row['class_id'])
        base = clean_lookup[key]

        drops.append({
            'precision_drop': base['precision'] - row['precision'],
            'recall_drop': base['recall'] - row['recall'],
            'mAP50_drop': base['mAP50'] - row['mAP50'],
            'mAP50_95_drop': base['mAP50_95'] - row['mAP50_95'],
        })

    drop_df = pd.DataFrame(drops)
    df = pd.concat([df, drop_df], axis=1)

    save_all = os.path.join(OUT_DIR, 'perclass_robustness_4cls.csv')
    df.to_csv(save_all, index=False, encoding='utf-8-sig')

    # 单独导出 severe_caries
    severe_df = df[df['class_name'] == 'severe_caries'].copy()
    save_severe = os.path.join(OUT_DIR, 'severe_caries_robustness.csv')
    severe_df.to_csv(save_severe, index=False, encoding='utf-8-sig')

    print('\n全部完成。')
    print(f'每类结果已保存：{save_all}')
    print(f'重度龋齿结果已保存：{save_severe}')

    print('\nsevere_caries 结果预览：')
    print(severe_df[['condition', 'precision', 'recall', 'mAP50', 'mAP50_95', 'recall_drop', 'mAP50_drop']])