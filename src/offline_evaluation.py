### 按照阿里提的标准计算错误率
import  numpy as np

GT_FILE = '/home/storage/lsy/fashion/FashionAI_Keypoint_Detection/r1_test_b' \
          '/fashionAI_key_points_test_b_answer_20180426.csv'  ### 真值文件
PRED_FILE = '/home/storage/lsy/fashion/kp_predictions/result.csv'### 预测的文件


def read_data(filename):
    input_file = open(filename, 'r')
    data_dict={}
    i = 0
    for line in input_file:
        if i == 0: ### drop  the header
            i = 1
            continue
        line = line.strip()
        line = line.split(',')
        name = line[0]
        type = line[1]
        def fn(x):
            c = x.split('_')
            return list(map(int,c[:]))
        joints = list(map(fn, line[2:]))
        joints = np.reshape(joints, (-1, 3))
        data_dict[name] = {'joints': joints,  'type': type}
    input_file.close()
    return data_dict

def calculate_norm(gt_data):
    samples = len(gt_data.keys())
    norm_mat = np.zeros((samples),np.float)
    for i,name  in enumerate(gt_data.keys()):
        catgory = gt_data[name]['type']
        pts = gt_data[name]['joints']
        if catgory == 'dress' or catgory == 'outwear' or catgory == 'blouse':
            norm = np.sqrt(np.square(pts[5][0] - pts[6][0]) + np.square(pts[5][1] - pts[6][1]))
        else:
            norm = np.sqrt(np.square(pts[15][0] - pts[16][0]) + np.square(pts[15][1] - pts[16][1]))
        if np.isnan(norm):
            print(' GT file not correct,  norm dis is NaN')
            exit(0)
        if norm==0:
           norm=256
        norm_mat[i] = norm
    return norm_mat

def calculate_norm_distance_mat(gt_data, pred_data,norm):
    samples = len(gt_data.keys())
    dis_mat = np.zeros((samples,24))
    n=0
    n_every_joints=np.zeros(24)
    for i,name in enumerate(gt_data.keys()):
        for j in range(24):
            # if gt_data[name]['joints'][j][2] != -1:
            if gt_data[name]['joints'][j][2]==1:## only visible

                n+=1
                n_every_joints[j]+=1
                gt_pts = gt_data[name]['joints'][j]
                pre_pts = pred_data[name]['joints'][j]
                d =np.sqrt((gt_pts[0]-pre_pts[0])*(gt_pts[0]-pre_pts[0]) + (gt_pts[1]-pre_pts[1])*(gt_pts[1]-pre_pts[1]) )
                dis_mat[i,j] = d/norm[i]
    return  dis_mat,n,n_every_joints


if __name__=='__main__':
    gt_data = read_data(GT_FILE)
    pre_data = read_data(PRED_FILE)
    samples = len(gt_data.keys())
    norm = calculate_norm(gt_data)
    norm_dis,N,n_every_joints = calculate_norm_distance_mat(gt_data, pre_data,norm)
    print(norm_dis.shape)
    err = np.sum(norm_dis)/N
    print('err: ', err*100)

    err_joints = np.sum(norm_dis,axis=0)
    err_joints = np.divide(err_joints,n_every_joints)*100
    for i,v in enumerate(err_joints):
        print('joints '+str(i)+' mean err: ', v)