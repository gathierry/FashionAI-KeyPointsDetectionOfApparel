import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2

class KPDA():
    def __init__(self, config, data_dir, train_val):
        self.config = config
        if train_val == 'test':
            data_dir += 'r2_test_b/'
            anno_df = pd.read_csv(data_dir + 'test.csv')
            anno_df['image_path'] = data_dir + anno_df['image_id']
        # elif train_val == 'val':
        #     data_dir += 'r1_test_b/'
        #     anno_df = pd.read_csv(data_dir + 'fashionAI_key_points_test_b_answer_20180426.csv')
        #     anno_df['image_path'] = data_dir + anno_df['image_id']
        else:
            data_dir0 = data_dir + 'wu_train/'
            anno_df0 = pd.read_csv(data_dir0 + 'Annotations/annotations.csv')
            anno_df0['image_path'] = data_dir0 + anno_df0['image_id']
            data_dir1 = data_dir + 'r1_train/'
            anno_df1 = pd.read_csv(data_dir1 + 'Annotations/train.csv')
            anno_df1['image_path'] = data_dir1 + anno_df1['image_id']
            data_dir2 = data_dir + 'r1_test_a/'
            anno_df2 = pd.read_csv(data_dir2 + 'fashionAI_key_points_test_a_answer_20180426.csv')
            anno_df2['image_path'] = data_dir2 + anno_df2['image_id']
            # anno_df = pd.concat([anno_df0, anno_df1, anno_df2])

            data_dir3 = data_dir + 'r1_test_b/'
            anno_df3 = pd.read_csv(data_dir3 + 'fashionAI_key_points_test_b_answer_20180426.csv')
            anno_df3['image_path'] = data_dir3 + anno_df3['image_id']
            anno_df3_train, anno_df3_val = train_test_split(anno_df3, test_size=0.2, random_state=42)
            if train_val == 'train':
                anno_df = pd.concat([anno_df0, anno_df1, anno_df2, anno_df3_train])
            else:
                anno_df = anno_df3_val


        self.anno_df = anno_df[anno_df['image_category'] == self.config.clothes]


    def size(self):
        return len(self.anno_df)

    def get_image_path(self, image_index):
        row = self.anno_df.iloc[image_index]
        image_path = row['image_path']
        return image_path

    def get_bbox(self, image_index, extend=10):
        row = self.anno_df.iloc[image_index]
        locs = []
        for key, item in row.iteritems():
            if key in self.config.keypoints[self.config.clothes]:
                loc = [int(x) for x in item.split('_')]
                if loc[0] != -1 and loc[1] != -1:
                    locs.append(loc[:2])
        locs = np.array(locs)
        minimums = np.amin(locs, axis=0)
        maximums = np.amax(locs, axis=0)
        bbox = np.array([[max(minimums[0]-extend, 0), max(minimums[1]-extend, 0),
                          maximums[0]+extend, maximums[1]+extend]], dtype=np.float32)
        return bbox

    def get_keypoints(self, image_index):
        row = self.anno_df.iloc[image_index]
        locs = []
        for key, item in row.iteritems():
            if key in self.config.keypoints[self.config.clothes]:
                loc = [int(x) for x in item.split('_')]
                locs.append(loc)
        locs = np.array(locs)
        return locs

def get_default_xy(config, db_path):
    kpda = KPDA(config, db_path, 'all')
    s = []
    for k in range(kpda.size()):
        path = kpda.get_image_path(k)
        img = cv2.imread(path)
        h, w, _ = img.shape
        locs = kpda.get_keypoints(k).astype(np.float32)
        locs[:, 0] = locs[:, 0].astype(np.float32) / float(w)
        locs[:, 1] = locs[:, 1].astype(np.float32) / float(h)
        locs[:, 2] = (locs[:, 2]>=0)*1.0
        s.append(locs)
    s = np.stack(s)
    s = s.sum(axis=0)
    s = s[:, :2] / s[:, 2:].repeat(2, axis=1)
    print(s)

if __name__ == '__main__':
    from tqdm import tqdm
    import cv2
    from src.config import Config
    config = Config('dress')
    kpda = KPDA(config, '/home/storage/lsy/fashion/FashionAI_Keypoint_Detection/', 'train')
    print(kpda.anno_df)



