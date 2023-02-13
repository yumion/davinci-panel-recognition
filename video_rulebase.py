from pathlib import Path
import cv2
import pandas as pd
from tqdm import trange
from recognizer import recognizer
from rulebase import camera_move_flag, third_arm_flag, stitch_flag, cut_coag_seal_flag, power_arm_flag


def main():
    video_p = Path('Video/047410018.mp4')
    save_dir = Path('./prediction')

    start_frame = 17400
    end_frame = 17600

    video = cv2.VideoCapture(str(video_p))
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if end_frame < 0 else end_frame

    save_dir = save_dir / video_p.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame()
    with trange(start_frame, end_frame) as pbar:
        for i in pbar:
            ret, frame = video.read()
            boxes = recognizer(frame)
            # 認識結果からルールベースでフラグを立てる
            result = {
                'frame': f'{i:06d}',
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

            pbar.set_postfix(result)
            result = pd.Series(result)
            results = pd.concat([results, result], axis=1)
    results = results.T  # 列名が行にあるので転置
    results.to_csv(save_dir / f'results_{video_p.stem}.csv', index=False)


if __name__ == '__main__':
    main()
