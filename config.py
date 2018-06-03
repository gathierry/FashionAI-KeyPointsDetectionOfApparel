import math
import numpy as np

class Config:

    def __init__(self, clothes):
        self.clothes = clothes
        self.keypoints = {'blouse' : ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                                      'armpit_left', 'armpit_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                                      'cuff_right_out', 'top_hem_left', 'top_hem_right'],
                          'outwear' : ['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'armpit_left',
                                       'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out',
                                       'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right'],
                          'trousers' : ['waistband_left', 'waistband_right', 'crotch', 'bottom_left_in', 'bottom_left_out',
                                        'bottom_right_in', 'bottom_right_out'],
                          'skirt' : ['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right'],
                          'dress' : ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                                     'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                                     'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'hemline_left', 'hemline_right']}
        keypoint = self.keypoints[self.clothes]
        self.num_keypoints = len(keypoint)
        self.conjug = []
        for i, key in enumerate(keypoint):
            if 'left' in key:
                j = keypoint.index(key.replace('left', 'right'))
                self.conjug.append([i, j])
        if self.clothes in ['outwear', 'blouse', 'dress']:
            self.datum = [keypoint.index('armpit_left'), keypoint.index('armpit_right')]
        elif self.clothes in ['trousers', 'skirt']:
            self.datum = [keypoint.index('waistband_left'), keypoint.index('waistband_right')]
        # Img
        self.img_max_size = 512
        self.mu = 0.65
        self.sigma = 0.25
        # RPN
        self.anchor_areas = [32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.]  # p3 -> p7
        self.aspect_ratios = [1 / 5., 1 / 2., 1 / 1., 2 / 1., 5 / 1.]  # w/h
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_num = len(self.aspect_ratios) * len(self.scale_ratios)
        self.max_iou = 0.7
        self.min_iou = 0.4
        self.cls_thresh = 0.5
        self.nms_thresh = 0.5
        self.nms_topk = 1

        # STAGE 2
        self.hm_stride = 4
        # Heatmap
        # if self.clothes in ['outwear', 'trousers']:
        #     self.hm_sigma = self.img_max_size / self.hm_stride / 8.
        # else:
        self.hm_sigma = self.img_max_size / self.hm_stride / 16. #4 #16 for 256 size
        self.hm_alpha = 100.

        lrschedule = {'blouse' : [16, 26, 42],
                      'outwear' : [15, 20, 26],
                      'trousers' : [18, 25, 36],
                      'skirt' : [26, 32, 39],
                      'dress' : [30, 34, 31]
                     }
        self.lrschedule = lrschedule[clothes]


if __name__ == '__main__':
    config = Config('outwear')
    print(config.conjug)