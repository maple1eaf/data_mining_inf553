import os
import json
import rmse

TUNING_FILE = "/Users/markduan/duan/USC_course/USC_APDS/INF553/project/predict/tuning.json"

CORATED_LIMIT = [3, 5, 7, 10]
LONELY_THRESHOLD = [2, 3, 5, 7]
N_NEIGHBORS_ITEMBASED = [5, 7, 10, 12]
WEIGHT = [0.2, 0.4, 0.6, 0.8]

def writeRes(c, l, n, w, res):
    with open(TUNING_FILE, 'a', encoding='utf-8') as fp:
        x = {
            'c': c,
            'l': l,
            'n': n,
            'w': w,
            'rmse': res
        }
        x_j = json.dumps(x)
        fp.write(x_j)
        fp.write('\n')

if os.path.exists(TUNING_FILE):
    os.remove(TUNING_FILE)

for c in CORATED_LIMIT:
    for l in LONELY_THRESHOLD:
        train_comm = "spark-submit train.py %d %d %d" % (c, l, l)
        os.system(train_comm)
        for n in N_NEIGHBORS_ITEMBASED:
            for w in WEIGHT:
                test_comm = "spark-submit predict.py %d %f" % (n, w)
                os.system(test_comm)
                res = rmse.getRmse()
                writeRes(c, l, n, w, res)


