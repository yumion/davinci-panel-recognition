import cv2
import numpy as np


def get_img(img_p):
    img = cv2.imread(str(img_p))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_bgr_hist(bgr):
    bgr_hist = {}
    # BGRヒストグラム取得
    colors = ['b', 'g', 'r']
    for i, ch in enumerate(colors):
        hist = cv2.calcHist([bgr], [i], None, [256], [0, 256])
        bgr_hist[ch] = hist
    return bgr_hist


def is_camera_active(
        bgr,
        B_range=(50, 255),
        B_th=500):
    # 左・中央・右の3分割画像の内、左の画像に対してのみ有効
    # cameraマークをさらに切り取る
    camera_bgr = bgr
    # cameraマークに対してヒストグラム閾値処理
    color_hist = get_bgr_hist(bgr)
    # Bの指定範囲にあるピクセル数の合計が閾値以上ならActiveと判断する
    B_max = sum(color_hist['b'][B_range[0]:B_range[1]])
    if (B_max >= B_th):
        return True
    else:
        return False


def is_active(bgr, th=100):
    # 左・中央・右の3分割画像の内、左の画像に対してのみ有効
    # B channel - R channelの画像において、黒い領域の面積が閾値以下なら
    # Activeと判断する
    b, g, r = cv2.split(bgr)
    # 輝度値0~10と200~255のピクセル数をカウント
    area = np.sum(((b - r) < 10) + ((b - r) > 200))
    if area < th:
        return True
    else:
        return False


# R: 220-255の範囲でmaxが200pix以上
# G: 180-230の範囲でmaxが200pix以上
def is_cut(
        color_hist,
        R_range=(220, 255),
        G_range=(180, 230),
        R_th=200,
        G_th=200):
    # 左・中央・右の3分割画像の内、右の画像に対してのみ有効
    # 各色の指定範囲にあるピクセル数の最大値が閾値以上ならCUT通電と判断する
    R_max = np.max(color_hist['r'][R_range[0]:R_range[1]])
    G_max = np.max(color_hist['g'][G_range[0]:G_range[1]])
    if (R_max >= R_th) and (G_max >= G_th):
        return True
    else:
        return False


# G: 60-110の範囲でmaxが200pix以上
# B: 130-180の範囲でmaxが200pix以上
def is_coag(
        color_hist,
        G_range=(60, 110),
        B_range=(130, 180),
        G_th=200,
        B_th=200):
    # 左・中央・右の3分割画像の内、右の画像に対してのみ有効
    # 各色の指定範囲にあるピクセル数の最大値が閾値以上ならCUT通電と判断する
    G_max = np.max(color_hist['g'][G_range[0]:G_range[1]])
    B_max = np.max(color_hist['b'][B_range[0]:B_range[1]])
    if (B_max >= B_th) and (G_max >= G_th):
        return True
    else:
        return False


def power_device_recognizer(bgr, if_not=None):
    color_hist = get_bgr_hist(bgr)
    if is_cut(color_hist):
        return 'cut'
    elif is_coag(color_hist):
        return 'coag'
    else:
        return if_not
