"""
modify DATA_COLLECTION_FOLDER & OUTPUT_FOLDER to your own exist folder path, and end up with '/'.
command examples:
python hw5test.py test 1,2,3,4,5
python hw5test.py test 1,3,5
python hw5test.py nmi 1,3,5
"""
import sys
import os
import json
import time

from sklearn.metrics.cluster import normalized_mutual_info_score

DATA_COLLECTION_FOLDER = "file:///Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw5/data/"
DATA_COLLECTION_FOLDER_NMI = "/Users/markduan/duan/USC_course/USC_APDS/INF553/homework/hw5/data/"
OUTPUT_FOLDER = "./output/"

service = sys.argv[1]
if service != 'help':
    cases_str = sys.argv[2]
    cases = json.loads("[" + cases_str + "]")

def generateCommand(case):
    i = case
    if i == 1:
        n_cluster = 10
    elif i == 2:
        n_cluster = 10
    elif i == 3:
        n_cluster = 5
    elif i == 4:
        n_cluster = 8
    elif i == 5:
        n_cluster = 15
    else:
        sys.exit()
    
    input_folder = DATA_COLLECTION_FOLDER + 'test%d' % (i)

    output_file1 = OUTPUT_FOLDER + 'out_file1_case%d.json' % (i)
    output_file2 = OUTPUT_FOLDER + 'out_file2_case%d.csv' % (i)
    
    comm = 'spark-submit bfr_wsj.py %s %s %s %s' % (input_folder, n_cluster, output_file1, output_file2)
    return comm

def getFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        d = json.load(fp)
    m = [(int(k), d[k]) for k in d]
    m.sort(key=lambda x: x[0])
    res = [x[1] for x in m]
    return res

def computeNMI(cases):
    nmi_list = []
    for case in cases:
        ground_truth_path = DATA_COLLECTION_FOLDER_NMI + "cluster%d.json" % (case)
        predict_path  = OUTPUT_FOLDER + "out_file1_case%d.json" % (case)
        ground_truth = getFile(ground_truth_path)
        prediction = getFile(predict_path)
        nmi = normalized_mutual_info_score(ground_truth, prediction)
        nmi_list.append(nmi)
        print('case%d: NMI = %f' % (case, nmi))
    return nmi_list

def test():
    t1 = time.time()
    dur = []
    for case in cases:
        comm = generateCommand(case)
        print("COMMAND:", comm)
        t_s = time.time()
        os.system(comm)
        t_e = time.time()
        dur.append(t_e - t_s)
    t2 = time.time()
    for i, c in enumerate(cases):
        print('\nrun case %d use %fs.' % (c, dur[i]))
    print('run cases use %fs.' % (t2 - t1))
    nmis = computeNMI(cases)

def nmi():
    nmis = computeNMI(cases)

if __name__ == "__main__":
    if service == 'test':
        test()
    elif service == 'nmi':
        nmi()
    elif service == 'help':
        print("""
modify DATA_COLLECTION_FOLDER & OUTPUT_FOLDER to your own exist folder path, and end up with '/'.
command examples:
python hw5test.py test 1,2,3,4,5
python hw5test.py test 1,3,5
python hw5test.py nmi 1,3,5
        """)
    else:
        print('please use "test" or "nmi" as the 1st parameter.')
        sys.exit()