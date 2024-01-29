import pandas as pd
import numpy as np
import csv
import datetime
import time

# %matplotlib inline
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics


DATA_ROOT_PATH = "/storage/nhatnt/ite_model_nhatnt/projects/input/site_data/"

retail_rocket_root_path = DATA_ROOT_PATH + 'retail_rocket/'
recobell_root_path = DATA_ROOT_PATH + "recobell/"
ml_1m_root_path = DATA_ROOT_PATH + "ml-1m/"

with open(recobell_root_path + 'i2index.txt') as f:
    active_item_dict = {str(line.split(',')[0]): int(line.split(',')[1]) for line in f}
active_item_list = active_item_dict.keys()
print(len(active_item_list))

with open(recobell_root_path + 'u2index.txt') as f:
    active_user_dict = {str(line.split(',')[0]): int(line.split(',')[1]) for line in f}
active_user_list = active_user_dict.keys()
print(len(active_user_list))

sorted_filtered_data = pd.read_csv(recobell_root_path + "sorted_filtered_data", header=None,
                                   names=["uid", "itemid", "time", "label"])

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("create train, test data").getOrCreate()

sorted_filtered_df = spark.read.csv(recobell_root_path + "sorted_filtered_data",
                                    "uid INT, itemid INT, time LONG, label INT")
sorted_filtered_df.show()
sorted_filtered_df = sorted_filtered_df.rdd.map(lambda x: (x[0], [(x[1], x[2], x[3])])).reduceByKey(
    lambda v1, v2: v1 + v2)


def list_sort(v):
    v.sort(key=lambda e: e[1])
    return v


no_time_filtered_df = sorted_filtered_df.mapValues(list_sort).mapValues(
    lambda v: [[e[0] for e in v], [e[2] for e in v]])
print(no_time_filtered_df.take(1))

import progressbar
import scipy.sparse as sp


def load_interact_matrix(df, num_user, num_item):
    start = datetime.datetime.now()
    # Construct matrix
    #     mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)
    mat = []
    for row in progressbar.ProgressBar()(df.itertuples(index=False)):
        uid, itemid = int(row[0]), int(row[1])
        mat.append((uid, itemid))
    print("time load_interact_matrix: ", datetime.datetime.now() - start)
    return mat


interact_matrix = load_interact_matrix(sorted_filtered_data, len(active_user_list), len(active_item_list))
interact_matrix_bcast = spark.sparkContext.broadcast(interact_matrix)

import random


def build_train_test(x, window_size, interact_matrix, num_item):
    def get_neg_list(uid, s):
        get_neg_list = []
        for _ in range(s):
            j = random.randrange(num_item)
            while (uid, j) in interact_matrix:
                j = random.randrange(num_item)
            get_neg_list.append(j)
        return get_neg_list

    res = []
    user_id = x[0]
    item_list = x[1][0]
    label_list = x[1][1]
    test_index = len(label_list) - 1
    while test_index >= 0:
        if label_list[test_index] != 1:
            break
        test_index -= 1
    # khong co addtocart va purchase
    if test_index < 0:
        # for training
        for i in range(len(item_list)):
            if i < window_size:
                t = [True,
                     user_id,
                     (window_size - i) * [-1] + item_list[:i],
                     (window_size - i) * [-1] + get_neg_list(user_id, i),
                     item_list[i],
                     label_list[i],
                     ]
            else:
                t = [True,
                     user_id,
                     item_list[i - window_size:i],
                     get_neg_list(user_id, window_size),
                     item_list[i],
                     label_list[i]
                     ]
            res.append(t)

    # co addtocart hoac purchase
    else:
        # for training
        for i in range(test_index):
            if i < window_size:
                t = [True,
                     user_id,
                     (window_size - i) * [-1] + item_list[:i],
                     (window_size - i) * [-1] + get_neg_list(user_id, i),
                     item_list[i],
                     label_list[i]
                     ]
            else:
                t = [True,
                     user_id,
                     item_list[i - window_size:i],
                     get_neg_list(user_id, window_size),
                     item_list[i],
                     label_list[i]
                     ]
            res.append(t)

        # for training
        new_start_point = test_index + 1
        for i in range(new_start_point, len(item_list)):
            if i - new_start_point < window_size:
                t = [True,
                     user_id,
                     (window_size - i) * [-1] + item_list[new_start_point:i],
                     (window_size - i) * [-1] + get_neg_list(user_id, i - new_start_point),
                     item_list[i],
                     label_list[i]
                     ]
            else:
                t = [True,
                     user_id,
                     item_list[i - window_size:i],
                     get_neg_list(user_id, window_size),
                     item_list[i],
                     label_list[i]
                     ]
                res.append(t)

        # for testing
        if test_index < window_size:
            t = [False,
                 user_id,
                 (window_size - test_index) * [-1] + item_list[:test_index],
                 item_list[test_index],
                 label_list[test_index]
                 ]
        else:
            t = [False,
                 user_id,
                 item_list[test_index - window_size:test_index],
                 item_list[test_index],
                 label_list[test_index]
                 ]
        res.append(t)

    #     # for training
    #     for i in range(len(item_list)):
    #         if i < window_size:
    #             t = [True,
    #                  user_id,
    #                  (window_size - i) * [-1] + item_list[:i],
    #                  (window_size - i) * [-1] + get_neg_list(user_id, i),
    #                  item_list[i],
    #                  label_list[i]
    #                 ]
    #         else:
    #             t = [True,
    #                  user_id,
    #                  item_list[i - window_size:i],
    #                  get_neg_list(user_id, window_size),
    #                  item_list[i],
    #                  label_list[i]
    #                 ]
    #         res.append(t)
    return res


test_final_df = no_time_filtered_df
num_item = len(active_item_list)
final_df = test_final_df.repartition(56).flatMap(
    lambda x: build_train_test(x, 5, interact_matrix_bcast.value, num_item))
import json

train_data_rdd = final_df.filter(lambda x: x[0]).map(lambda x: [x[1], json.dumps(x[2]), json.dumps(x[3]), x[4], x[5]])
train_data_df = train_data_rdd.toDF(["user_id", "last_item_ids", "last_item_ids_neg", "target_item_id", "target_label"])
train_data_df.show(truncate=False)
train_data_df.write.mode("overwrite").csv(recobell_root_path + "sequence_train_data", quote="|", quoteAll=True)

spark.sparkContext.textFile()