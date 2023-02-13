import numpy as np
from PIL import Image
import difflib
import pyocr


CATEGORIES = [
    'maryland bipolar forceps',
    'monopolar curved scissors',
    'vessel sealer extend',
    'large needle driver',
    'prograsp forceps',
]


def bitwise(gray: np.ndarray, th: int = 0):
    '''GRAY画像の2値化
    input: GRAY
    output: GRAY
    '''
    bw = np.zeros_like(gray)
    bw[gray > th] = 255
    return bw


def search_words(
        gray: np.ndarray,
        replace_char: dict = {'\n': ' ', '.': ''}):
    '''
    gray画像の中の英文字を認識する
    認識したい文字以外は画像から除くと精度が高い
    '''
    # NOTE: 2値化した方がOCRの精度が良い
    # # 大津の二値化で適応的に閾値を決める
    # th, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 90パーセンタイルの輝度値を閾値とする
    th = int(np.percentile(gray.flatten(), q=[90])[0])
    bw = bitwise(gray, th)
    ocr = pyocr.get_available_tools()[0]
    res = ocr.image_to_string(
        Image.fromarray(bw),
        lang='eng',
        builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    ).lower()  # 全て小文字にする
    # 改行やピリオドなどを取り除く
    for k, v in replace_char.items():
        res = res.replace(k, v)
    return res


def match_word_score(source: str, target: str):
    '''単語の一致度を計算する'''
    return difflib.SequenceMatcher(
        a=source,
        b=target,
        isjunk=None,
        autojunk=True).ratio()


def decision_device(name, th_rate=0.9):
    '''認識文字列の中で、類似スコアが閾値以上で最も高い術具に決定する
    閾値未満の場合は画面に文字がない可能性が高いので空文字を返す
    '''
    scores = [match_word_score(name, cat) for cat in CATEGORIES]
    if th_rate < max(scores):
        return CATEGORIES[np.argmax(scores)]
    return ''
