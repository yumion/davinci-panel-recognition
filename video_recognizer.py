from pathlib import Path
from tqdm import trange
import cv2
import pandas as pd
from recognizer import recognizer


if __name__ == '__main__':
    video_p = Path('Video/022410271.mp4')
    save_dir = Path('./prediction')

    start_frame = 17400
    end_frame = 17600

    video = cv2.VideoCapture(str(video_p))
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = video.get(cv2.CAP_PROP_FRAME_COUNT) if end_frame < 0 else end_frame

    save_dir = save_dir / video_p.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame()
    with trange(start_frame, end_frame) as pbar:
        for i in pbar:
            ret, frame = video.read()
            boxes = recognizer(frame)

            result = {
                'db': video_p.stem,
                'frame': f'{i:06d}',
            }
            for i, box in boxes.items():
                result[f'box{i}'] = box.get('name')
                result[f'state{i}'] = box.get('state')
            pbar.set_postfix(result)
            result = pd.Series(result)
            results = pd.concat([results, result], axis=1)

    results = results.T  # 列名が行にあるので転置
    results.to_csv(save_dir / f'results_{video_p.stem}_{start_frame}_{end_frame}.csv', index=False)
