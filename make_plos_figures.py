import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# =========================
# 只改这里
# =========================
FIG_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/robust_summary_4cls/figures'
MONTAGE_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/qualitative_examples_4cls/montage'

OUT_DIR = r'C:/Users/qq235/Desktop/V8/YOLOv8-yan/runs/plos_figures'

FIG1_A = 'figure_overall_map50_drop.png'
FIG1_B = 'figure_overall_map5095_drop.png'

FIG2_A = 'figure_severe_recall_degradation.png'
FIG2_B = 'figure_severe_map50_degradation.png'

FIG3_SOURCE = '1_montage.jpg'
# =========================


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def load_image(path):
    img = Image.open(path).convert('RGB')
    return img


def resize_to_width(img, width):
    w, h = img.size
    scale = width / w
    new_h = int(h * scale)
    return img.resize((width, new_h), Image.LANCZOS)


def get_font(size=42):
    """
    Windows 默认字体。
    如果找不到，就使用 PIL 默认字体。
    """
    font_candidates = [
        r'C:/Windows/Fonts/arialbd.ttf',
        r'C:/Windows/Fonts/arial.ttf',
    ]

    for fp in font_candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)

    return ImageFont.load_default()


def add_panel_label(img, label):
    """
    在左上角加 A/B 标签。
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = get_font(48)

    # 白底标签框
    box_w, box_h = 80, 65
    draw.rectangle([0, 0, box_w, box_h], fill='white')
    draw.text((18, 8), label, fill='black', font=font)

    return img


def combine_two_panels(img_a_path, img_b_path, out_path, panel_width=1800, gap=80):
    """
    左右合并两张图，添加 A/B 面板标签。
    """
    img_a = load_image(img_a_path)
    img_b = load_image(img_b_path)

    img_a = resize_to_width(img_a, panel_width)
    img_b = resize_to_width(img_b, panel_width)

    img_a = add_panel_label(img_a, 'A')
    img_b = add_panel_label(img_b, 'B')

    h = max(img_a.size[1], img_b.size[1])
    w = img_a.size[0] + img_b.size[0] + gap

    canvas = Image.new('RGB', (w, h), 'white')
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (img_a.size[0] + gap, 0))

    canvas.save(out_path, dpi=(300, 300), quality=95)
    print('已保存:', out_path)


def prepare_fig3(src_path, out_path, target_width=3600):
    """
    处理 qualitative montage，统一输出为 Fig3。
    """
    img = load_image(src_path)

    if img.size[0] > target_width:
        img = resize_to_width(img, target_width)

    img.save(out_path, dpi=(300, 300), quality=95)
    print('已保存:', out_path)


if __name__ == '__main__':
    mkdir(OUT_DIR)

    fig1_a_path = Path(FIG_DIR) / FIG1_A
    fig1_b_path = Path(FIG_DIR) / FIG1_B

    fig2_a_path = Path(FIG_DIR) / FIG2_A
    fig2_b_path = Path(FIG_DIR) / FIG2_B

    fig3_src_path = Path(MONTAGE_DIR) / FIG3_SOURCE

    print('检查 Fig1 A:', fig1_a_path.exists(), fig1_a_path)
    print('检查 Fig1 B:', fig1_b_path.exists(), fig1_b_path)
    print('检查 Fig2 A:', fig2_a_path.exists(), fig2_a_path)
    print('检查 Fig2 B:', fig2_b_path.exists(), fig2_b_path)
    print('检查 Fig3:', fig3_src_path.exists(), fig3_src_path)

    if not fig1_a_path.exists():
        raise FileNotFoundError(fig1_a_path)
    if not fig1_b_path.exists():
        raise FileNotFoundError(fig1_b_path)
    if not fig2_a_path.exists():
        raise FileNotFoundError(fig2_a_path)
    if not fig2_b_path.exists():
        raise FileNotFoundError(fig2_b_path)
    if not fig3_src_path.exists():
        raise FileNotFoundError(fig3_src_path)

    fig1_out = Path(OUT_DIR) / 'Fig1_overall_robustness.png'
    fig2_out = Path(OUT_DIR) / 'Fig2_severe_caries_robustness.png'
    fig3_out = Path(OUT_DIR) / 'Fig3_qualitative_examples.png'

    combine_two_panels(fig1_a_path, fig1_b_path, fig1_out)
    combine_two_panels(fig2_a_path, fig2_b_path, fig2_out)
    prepare_fig3(fig3_src_path, fig3_out)

    print('\n全部完成。')
    print('PLOS 图片输出目录:', OUT_DIR)