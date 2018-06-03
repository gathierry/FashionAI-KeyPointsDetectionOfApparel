import cv2
import pandas as pd
import numpy as np

from config import Config
from kpda_parser import KPDA
from utils import draw_keypoint_with_caption

root_path = '/home/storage/lsy/fashion/'
db_path = root_path + 'FashionAI_Keypoint_Detection/'

if __name__ == '__main__':
    config = Config('blouse')
    test_kpda = KPDA(config, db_path, 'test')
    df = pd.read_csv(root_path + 'kp_predictions/' + config.clothes + '.csv')
    for idx in range(3):
        img_path = test_kpda.get_image_path(idx)
        img0 = cv2.imread(img_path)  # BGR
        row = test_kpda.anno_df.iloc[idx]
        row2 = df[df['image_id'] == row['image_id']].T.squeeze()
        hps = []
        for k, v in row2.iteritems():
            if k in ['image_id', 'image_category'] or v.split('_')[2] == '-1':
                continue
            x = int(v.split('_')[0])
            y = int(v.split('_')[1])
            image = draw_keypoint_with_caption(img0, [x, y], k)
            hps.append(image)
        cv2.imwrite('/home/storage/lsy/fashion/tmp/%d.png' % idx, np.concatenate(hps, axis=1))
