"""写真なしで試せる、CGの乾麺サンプル画像を作るスクリプト。"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / 'sample_images'


def generate_images(out_dir=OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 12本を重ならずに配置した単純な模擬画像。
    spread = np.zeros((860, 1020, 3), dtype=np.uint8)
    spread[:] = (27, 33, 38)
    for i in range(12):
        y = 90 + i * 58
        cv2.line(spread, (120, y), (920, y), (244, 207, 101), 9, cv2.LINE_8)
    Image.fromarray(spread).save(out_dir / 'demo_spread_12.png')

    # 16本の断面模擬画像。実際のパスタ束の写真ではない。
    bundle = np.zeros((680, 680, 3), dtype=np.uint8)
    bundle[:] = (28, 33, 39)
    for i in range(4):
        for j in range(4):
            x, y = 180 + i * 78, 180 + j * 78
            cv2.circle(bundle, (x, y), 19, (239, 198, 101), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(bundle, (x, y), 19, (253, 220, 136), thickness=2, lineType=cv2.LINE_AA)
    Image.fromarray(bundle).save(out_dir / 'demo_bundle_16.png')
    return out_dir


if __name__ == '__main__':
    print('Demo images saved to:', generate_images())
