import json

import numpy as np
from sklearn.metrics import mean_squared_error

GROUND_TRUTH_PATH = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/data/test_review_ratings.json"
PREDICTION_PATH = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/predict/prediction_combine.json"

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def getYList(file_path):
    gt = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
#     print(len(lines))
    for line in lines:
        line = line[:-1]
        r_dict = json.loads(line)
        uid = r_dict['user_id']
        bid = r_dict['business_id']
        stars = r_dict['stars']
        gt.append(((uid, bid), stars))
#     print(len(gt))
    return gt

def getY(y_list):
    y = []
    for i in range(len(y_list)):
        y.append(y_list[i][1])
    return y

def getYReduce(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print('diff lenght!')
        return None, None
    miss = 0
    y_true_reduce = []
    y_pred_reduce = []
    for i in range(len(y_true)):
        if y_pred[i] == None:
            miss += 1
        else:
            y_true_reduce.append(y_true[i])
            y_pred_reduce.append(y_pred[i])
    print('miss %d values.' % (miss))
    return y_true_reduce, y_pred_reduce     

def allPredict(pre_file_path):
    gt_file_path = GROUND_TRUTH_PATH
    gt_list = getYList(gt_file_path)

    pre_list = getYList(pre_file_path)

    y_true = getY(gt_list)
    y_pred = getY(pre_list)

    res = rmse(y_true, y_pred)
    
    return res

def withNone(pre_file_path):
    gt_file_path = GROUND_TRUTH_PATH
    gt_list = getYList(gt_file_path)

    pre_list = getYList(pre_file_path)

    y_true = getY(gt_list)
    y_pred = getY(pre_list)
    y_true_reduce, y_pred_reduce = getYReduce(y_true, y_pred)

    res = rmse(y_true_reduce, y_pred_reduce)
    
    return res

def getRmse():
    res = withNone(PREDICTION_PATH)
    return res

if __name__ == "__main__":
    getRmse()
