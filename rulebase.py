from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
from recognizer import recognizer


def camera_move_flag(boxes):
    '''cameraがactiveであればcamera移動'''
    for box in boxes.values():
        if box.get('name') == 'camera':
            return 1
    return 0


def third_arm_flag(boxes):
    '''prograsp forcepsがactiveであれば3rd arm'''
    for box in boxes.values():
        if box.get('name') == 'prograsp forceps':
            return 1
    return 0


def stitch_flag(boxes):
    '''large needle driverが2本activeであれば縫合'''
    cnt = 0
    for box in boxes.values():
        if box.get('name') == 'large needle driver':
            cnt += 1
    if cnt >= 2:
        return 1
    return 0


def cut_coag_seal_flag(boxes):
    power_devices = ['maryland bipolar forceps',
                     'monopolar curved scissors',
                     'vessel sealer extend']
    power_category = ['cut', 'coag']
    for box in boxes.values():
        if box.get('name') in power_devices:
            if box['name'] == 'vessel sealer extend' and box['state'] == 'coag':
                return 'seal'
            elif box['state'] in power_category:
                return box['state']


def power_arm_flag(boxes):
    power_category = ['cut', 'coag']
    for box in boxes.values():
        if box.get('state') in power_category:
            return box['name']


def main():
    case_dir = Path('Image/047410018')
    save_dir = Path('./prediction')

    save_dir = save_dir / case_dir.name
    save_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame()
    with tqdm(case_dir.glob('movieframe/*.png')) as pbar:
        for frame_p in pbar:
            frame = cv2.imread(str(frame_p))
            boxes = recognizer(frame)
            # 認識結果からルールベースでフラグを立てる
            result = {
                'db': case_dir.name,
                'frame': frame_p.stem.replace('frame_', ''),
                'camera_move': camera_move_flag(boxes),
                'third_arm': third_arm_flag(boxes),
                'stitch': stitch_flag(boxes),
                'power_category': cut_coag_seal_flag(boxes),
                'power_arm': power_arm_flag(boxes),
            }
            # 加工前の結果も列に追加する
            for i, box in boxes.items():
                result[f'box{i}'] = box.get('name')
                result[f'state{i}'] = box.get('state')

            # print(result)
            pbar.set_postfix(result)
            result = pd.Series(result)
            results = pd.concat([results, result], axis=1)
    results = results.T  # 列名が行にあるので転置
    results.to_csv(save_dir / f'analysis_{case_dir.name}.csv', index=False)


if __name__ == '__main__':
    main()
