import os
from tqdm import tqdm
import cv2
import pandas as pd


'''術具状態表示座標は以下
番号: 左上(w, h), 右下(w, h)
1: (220, 664), ( 427, 696)
2: (431, 664), ( 637, 696)
3: (642, 664), ( 850, 696)
4: (854, 664), (1060, 696)
以下ではcv2の画像の配列に合わせて(h,w)で設定する'''
coord1 = [[664, 219], [696, 427]]
coord2 = [[664, 431], [696, 639]]
coord3 = [[664, 642], [696, 850]]
coord4 = [[664, 854], [696, 1062]]
COORDS = [coord1, coord2, coord3, coord4]


def crop_box(img, l_top, r_bottom):
    # imgのshapeは(h,w,c)、left_top, right_bottomは(h,w)
    return img[l_top[0]:r_bottom[0], l_top[1]:r_bottom[1], :]


def split_block(img):
    '''
    左: 番号;active判定に使用
    中央: 文字;
    右: CUT/COAG専用
    '''
    return img[:, :34, :], img[:, 35:154, :], img[:, 154:, :]


if __name__ == '__main__':
    df = pd.read_csv('davinci.csv')

    for db in df.db.unique():
        with tqdm(df.loc[df.db == db, 'frame']) as pbar:
            pbar.set_description(f'{db:09d}')
            video = cv2.VideoCapture(f'Video/{db:09d}.mp4')
            for i in pbar:
                i = int(i)
                os.makedirs(f'Image/{db:09d}/movieframe', exist_ok=True)
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                _, frame = video.read()
                cv2.imwrite(f'Image/{db:09d}/movieframe/frame_{i:06d}.png', frame)
                
                for j, coord in enumerate(COORDS):
                    os.makedirs(f'Image/{db:09d}/box{j+1}', exist_ok=True)
                    box = crop_box(frame, coord[0], coord[1])
                    cv2.imwrite(f'Image/{db:09d}/box{j+1}/frame_{i:06d}.png', box)
