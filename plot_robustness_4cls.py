import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 只改这里
# =========================
SUMMARY_CSV = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/robust_summary_4cls/robustness_summary_4cls.csv'
SEVERE_CSV = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/robust_summary_4cls/severe_caries_robustness.csv'
OUT_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/robust_summary_4cls/figures'
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

level_order = {
    'mild': 1,
    'medium': 2,
    'severe': 3,
}

main_degradations = [
    'color_shift',
    'contrast_shift',
    'low_light',
    'over_exposure',
    'occlusion',
    'motion_blur',
    'gaussian_blur',
    'jpeg_compression',
]


def parse_condition(condition):
    if condition == 'clean_test':
        return 'clean', 'clean'

    parts = condition.split('_')
    level = parts[-1]
    deg = '_'.join(parts[:-1])
    return deg, level


def prepare_df(df):
    df = df.copy()
    parsed = df['condition'].apply(parse_condition)
    df['degradation'] = [x[0] for x in parsed]
    df['level'] = [x[1] for x in parsed]
    df['level_order'] = df['level'].map(level_order).fillna(0)
    return df


# =========================
# 1. Overall mAP50 drop bar
# =========================
summary = pd.read_csv(SUMMARY_CSV)
summary = prepare_df(summary)

plot_df = summary[summary['condition'] != 'clean_test'].copy()
plot_df = plot_df[plot_df['degradation'].isin(main_degradations)]
plot_df = plot_df.sort_values(['degradation', 'level_order'])

plt.figure(figsize=(14, 6))
plt.bar(plot_df['condition'], plot_df['mAP50_drop'])
plt.xticks(rotation=75, ha='right')
plt.ylabel('mAP50 drop')
plt.xlabel('Degradation condition')
plt.title('Overall robustness under image degradation')
plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'figure_overall_map50_drop.png')
plt.savefig(save_path, dpi=300)
plt.close()
print('已保存:', save_path)


# =========================
# 2. Overall mAP50-95 drop bar
# =========================
plt.figure(figsize=(14, 6))
plt.bar(plot_df['condition'], plot_df['mAP50_95_drop'])
plt.xticks(rotation=75, ha='right')
plt.ylabel('mAP50-95 drop')
plt.xlabel('Degradation condition')
plt.title('Overall localization robustness under image degradation')
plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'figure_overall_map5095_drop.png')
plt.savefig(save_path, dpi=300)
plt.close()
print('已保存:', save_path)


# =========================
# 3. Severe caries recall line
# =========================
severe = pd.read_csv(SEVERE_CSV)
severe = prepare_df(severe)

selected = severe[
    severe['degradation'].isin(['color_shift', 'contrast_shift', 'occlusion', 'low_light', 'over_exposure'])
].copy()

plt.figure(figsize=(10, 6))

for deg in ['color_shift', 'contrast_shift', 'occlusion', 'low_light', 'over_exposure']:
    sub = selected[selected['degradation'] == deg].sort_values('level_order')
    plt.plot(sub['level'], sub['recall'], marker='o', label=deg)

plt.axhline(
    severe.loc[severe['condition'] == 'clean_test', 'recall'].values[0],
    linestyle='--',
    label='clean_test'
)

plt.ylabel('Recall of severe caries')
plt.xlabel('Degradation severity')
plt.title('Severe caries recall under image degradation')
plt.legend()
plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'figure_severe_recall_degradation.png')
plt.savefig(save_path, dpi=300)
plt.close()
print('已保存:', save_path)


# =========================
# 4. Severe caries mAP50 line
# =========================
plt.figure(figsize=(10, 6))

for deg in ['color_shift', 'contrast_shift', 'occlusion', 'low_light', 'over_exposure']:
    sub = selected[selected['degradation'] == deg].sort_values('level_order')
    plt.plot(sub['level'], sub['mAP50'], marker='o', label=deg)

plt.axhline(
    severe.loc[severe['condition'] == 'clean_test', 'mAP50'].values[0],
    linestyle='--',
    label='clean_test'
)

plt.ylabel('mAP50 of severe caries')
plt.xlabel('Degradation severity')
plt.title('Severe caries mAP50 under image degradation')
plt.legend()
plt.tight_layout()
save_path = os.path.join(OUT_DIR, 'figure_severe_map50_degradation.png')
plt.savefig(save_path, dpi=300)
plt.close()
print('已保存:', save_path)

print('\n全部图生成完成。')
print('输出目录:', OUT_DIR)