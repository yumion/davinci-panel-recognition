from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
from get_box import COORDS, crop_box, split_block
from ocr import bitwise, search_words, decision_device
from is_camera import is_camera
from color_recognizer import is_active, is_camera_active, power_device_recognizer


def bgr2gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def recognizer(frame):
    boxes = {}
    # 画面下部の4boxへ分割
    for i, coord in enumerate(COORDS, 1):
        box = {}
        # frameからboxを切り取り
        img = crop_box(frame, *coord)
        # box内を3blockに分割
        lb, mb, rb = split_block(img)

        if is_camera(bgr2gray(mb), 0.5):
            # カメラ認識
            box['name'] = 'camera'

        # カメラの場合active判定方法を変える
        if box.get('name') == 'camera':
            # cameraのロゴを入力する
            active = is_active(mb[20:28, 61:80], 50)
        else:
            active = is_active(lb, 100)
        # 各boxのstateを判定し、deactiveでは認識しない
        if not active:
            # 結果の保持
            boxes[i] = box
            continue

        # activeの場合、boxの中身を認識をする
        if box.get('name') != 'camera':
            # カメラではない場合、各boxの術具名を認識
            word = search_words(bgr2gray(mb))
            # print(word)
            box['name'] = decision_device(word, 0.5)

        # CUT/COAG認識
        box['state'] = power_device_recognizer(rb, 'active')

        # 結果の保持
        boxes[i] = box
    return boxes


if __name__ == '__main__':
    case_dir = Path('Image/047410018')
    save_dir = Path('./prediction/')

    save_dir = save_dir / case_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame()
    with tqdm(sorted(case_dir.glob('movieframe/*.png'))) as pbar:
        for frame_p in pbar:
            frame = cv2.imread(str(frame_p))
            boxes = recognizer(frame)

            result = {
                'db': case_dir.name,
                'frame': frame_p.stem.replace('frame_', ''),
            }
            for i, box in boxes.items():
                result[f'box{i}'] = box.get('name')
                result[f'state{i}'] = box.get('state')
            pbar.set_postfix(result)
            result = pd.Series(result)
            results = pd.concat([results, result], axis=1)
    results = results.T  # 列名が行にあるので転置
    results.to_csv(save_dir / f'results_{case_dir.name}.csv', index=False)
