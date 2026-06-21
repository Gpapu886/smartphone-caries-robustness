import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# =========================
# 只改这里
# =========================
WEIGHT_PATH = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/train/V8n_4cls_augtrain/weights/best.pt'
CLEAN_YAML = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_split_4cls_augtrain/caries_split_4cls_augtrain.yaml'
DEGRADED_ROOT = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/dataset_4cls_degraded_test'
# =========================


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

    model = YOLO(WEIGHT_PATH)

    results = []

    # =========================
    # 1. clean test
    # =========================
    print('\n正在验证 clean_test...')

    metrics = model.val(
        data=CLEAN_YAML,
        split='test',
        imgsz=640,
        batch=16,
        device='0',
        workers=0,
        plots=True,
        project='runs/robust_val_4cls',
        name='clean_test'
    )

    results.append({
        'condition': 'clean_test',
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
    })

    # =========================
    # 2. degraded test
    # =========================
    degraded_root = Path(DEGRADED_ROOT)
    degraded_sets = sorted([p for p in degraded_root.iterdir() if p.is_dir()])

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
            plots=True,
            project='runs/robust_val_4cls',
            name=condition
        )

        results.append({
            'condition': condition,
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'mAP50': float(metrics.box.map50),
            'mAP50_95': float(metrics.box.map),
        })

    df = pd.DataFrame(results)

    clean_map50 = df.loc[df['condition'] == 'clean_test', 'mAP50'].values[0]
    clean_map5095 = df.loc[df['condition'] == 'clean_test', 'mAP50_95'].values[0]
    clean_recall = df.loc[df['condition'] == 'clean_test', 'recall'].values[0]
    clean_precision = df.loc[df['condition'] == 'clean_test', 'precision'].values[0]

    df['mAP50_drop'] = clean_map50 - df['mAP50']
    df['mAP50_95_drop'] = clean_map5095 - df['mAP50_95']
    df['recall_drop'] = clean_recall - df['recall']
    df['precision_drop'] = clean_precision - df['precision']

    os.makedirs('runs/robust_summary_4cls', exist_ok=True)

    save_path = 'runs/robust_summary_4cls/robustness_summary_4cls.csv'
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print('\n全部验证完成。')
    print(df)
    print(f'\n已保存：{save_path}')