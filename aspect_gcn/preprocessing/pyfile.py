# lib
import csv
import random
import progressbar
import scipy.sparse as sp
import datetime

from src import settings
from src.data_preparation.sparse_vector import *


# return number of user, item.
def load_representation_data(u2index_path, i2index_path):
    u2index = {}
    i2index = {}

    count = 0
    with open(u2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_user = count
    print('Num user: ', num_user)

    count = 0
    with open(i2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_item = count
    print('Num item: ', num_item)

    return num_user, num_item


def load_representation_data_with_item_repr(u2index_path, i2index_path, item_repr_path):
    u2index = {}
    i2index = {}
    item_repr = {}

    count = 0
    with open(u2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_user = count
    print('Num user: ', num_user)

    count = 0
    with open(i2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_item = count
    print('Num item: ', num_item)

    count = 0

    with open(item_repr_path) as f:
        item_pcat_dimension = int(next(f))
        csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for row in csv_reader:
            count += 1
            item_repr[int(row[0])] = json_string_to_tensorflow_sparse_vector(row[1])
    print('Dimension of item pcat: ', item_pcat_dimension)
    print('Num item (check again): ', len(item_repr))

    return num_user, num_item, item_repr, item_pcat_dimension


def load_representation_data_with_both_user_item_repr(u2index_path, i2index_path, user_repr_path, item_repr_path):
    u2index = {}
    i2index = {}
    user_repr = {}
    item_repr = {}

    count = 0
    with open(u2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_user = count
    print('Num user: ', num_user)

    count = 0
    with open(i2index_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_reader:
            count += 1
    num_item = count
    print('Num item: ', num_item)

    count = 0

    with open(user_repr_path) as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for row in csv_reader:
            count += 1
            user_repr[int(row[0])] = json_string_to_tensorflow_sparse_vector(row[1])
    print('Num user have pcat: ', len(user_repr))
    count = 0

    with open(item_repr_path) as f:
        item_pcat_dimension = int(next(f))
        csv_reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_ALL)
        for row in csv_reader:
            count += 1
            item_repr[int(row[0])] = json_string_to_tensorflow_sparse_vector(row[1])

    print('Dimension of item pcat: ', item_pcat_dimension)
    print('Num item (check again): ', len(item_repr))

    return num_user, num_item, user_repr, item_repr, item_pcat_dimension


# file_path = scene_1/_explicit.train.rating
def load_interact_matrix(file_path, num_user, num_item):
    start = datetime.datetime.now()
    # Construct matrix
    # ma tran thua voi num_user x num_item , init value la false.
    mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            uid, itemid = int(line[0]), int(line[1])  # get uid and item id.
            mat[uid, itemid] = True
    print("time load_interact_matrix: ", datetime.datetime.now() - start)
    return mat
    # return matrix voi cac phan tu tuong ung co trong file explicit.train.rating la True
    # con lai la cac phan tu la False.


# file_path = ratings_train.txt
def load_interact_matrix_s_ite(file_path, num_user, num_item):
    start = datetime.datetime.now()
    # Construct matrix
    # ma tran thua voi num_user x num_item , init value la false.
    mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            uid = int(line[0])
            itemids = line[1].strip()[1:-1]
            itemids = itemids.split(",")
            for item in itemids:
                itemid = int(item.strip())
                mat[uid, itemid] = True
    print("time load_interact_matrix_s_ite: ", datetime.datetime.now() - start)
    return mat
    # return matrix voi cac phan tu tuong ung co trong file explicit.train.rating la True
    # con lai la cac phan tu la False.


# scene_1/_explicit.test.rating
def load_test_data(file_path):
    rating_list = []
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            user, item = int(line[0]), int(line[1])  # get uid, itemid
            rating_list.append([user, item])
    print('len test data: ', len(rating_list))
    return rating_list


# 25600,48637
# 12290,16529
# 4,2504
# 28677,17621
# 22627,23301
def load_test_data_s_ite(file_path):
    rating_list = []
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            user, item = int(line[0]), int(line[1])  # get uid, itemid
            rating_list.append([user, item])
    print('Length test data: ', len(rating_list))
    return rating_list


# load from: scene_1/_explicit.test.negative => return list negative item for this user.
def load_negative_data(file_path):
    negative_dict = {}
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        # i = 0
        # raw data: (4,2504)|15383|6979|41741|79116|53192|12932|29099|
        for line in progressbar.ProgressBar()(csv_reader):
            # line: ['(4,2504)', '15383', '6979', '41741', '79116', '53192', '12932',..]
            user = line[0].split(",")  # ['(158', '4966)']
            user = int(user[0][1:])  # 158
            # assert user == i
            # i += 1

            negative_dict[user] = []  # doi voi moi user co 999 item tuong ung.
            for x in line[1:]:  # danh sach cac item tuong ung voi user do khong co tuong tac explicit.
                negative_dict[user].append(int(x))
    print('len negative data: ', len(negative_dict))
    return negative_dict  # user_id: list negative item.


def get_train_instances(training_dict, num_negatives, num_user, num_item):
    user_input, item_input, labels, indicator = [], [], [], []
    start = datetime.datetime.now()
    set_total_items = set(range(num_item))
    for (uid, itemid) in progressbar.ProgressBar()(training_dict):
        is_ex = training_dict[(uid, itemid)]
        user_input.append(uid)
        item_input.append(itemid)
        labels.append(1)
        # im_indicator.append(1)
        if is_ex:
            indicator.append(1)
        else:
            indicator.append(0)

        # negative sample
        # Lay mau cac item chua tung view hay purchase; train_data_mt = 1__purchase, -1__view
        for _ in range(num_negatives):
            j = random.randrange(num_item)
            while (uid, j) in training_dict:
                j = random.randrange(num_item)
            # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
            user_input.append(uid)
            item_input.append(j)
            labels.append(0)
            indicator.append(0)
        # end for

    print("time get_train_instances: ", datetime.datetime.now() - start)
    return user_input, item_input, labels, indicator


# get train data to fit model (can thay doi).
# file_path: partitioned_train_data/part_000
# interact matrix (loading from _explicit.train.rating)
# num_negative: set from hyper parameter.
# num_user
# num_item.
def get_train_instances_partition(file_path, interact_mat, num_negatives, num_user, num_item):
    user_input, item_input, labels, y1_indicator, y2_indicator = [], [], [], [], []
    start = datetime.datetime.now()
    with open(file_path, "r") as f:  # partitioned_train_data/part_000 reading from ratings divition file.
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):

            # each row in _explicit.train.rating
            uid, itemid, explicit_timestamp = int(line[0]), int(line[1]), int(line[4])

            is_ex = explicit_timestamp > 0  # check time_stamp explicit > 0 => co explicit.
            user_input.append(uid)
            item_input.append(itemid)
            labels.append(1)
            # im_indicator.append(1)
            y1_indicator.append(1)
            y2_indicator.append(0)

            # negative instances for implicit.
            # Lay mau cac item chua tung view hay purchase; train_data_mt = 1__purchase, -1__view
            # negative sampling num_negatives items cho user uid.
            for _ in range(num_negatives):
                j = random.randrange(num_item)  # select number from 0 - num_item.
                while (uid, j) in interact_mat:
                    j = random.randrange(num_item)
                # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
                user_input.append(uid)
                item_input.append(j)  # j chua tung duoc tuong tac.
                labels.append(0)
                y1_indicator.append(1)
                y2_indicator.append(0)

            # truong hop dac biet neu cap tuong tac nay la explicit thi se training tiep tang thu 2 la explicit.
            # va tiep tuc sampling negative cho tuong tac explicit day.
            if is_ex:
                user_input.append(uid)
                item_input.append(itemid)
                labels.append(1)
                # im_indicator.append(1)
                y1_indicator.append(0)
                y2_indicator.append(1)

                # negative instances
                # Lay mau cac item chua tung view hay purchase; train_data_mt = 1__purchase, -1__view
                for _ in range(num_negatives):
                    j = random.randrange(num_item)
                    while (uid, j) in interact_mat:
                        j = random.randrange(num_item)
                    # Neu lay the nay co kha nang (u, j_) xuat hien hon 1 lan trong tap negative
                    user_input.append(uid)
                    item_input.append(j)
                    labels.append(0)
                    y1_indicator.append(0)
                    y2_indicator.append(1)
            # end for
    print("time get_train_instances: ", datetime.datetime.now() - start)
    return user_input, item_input, labels, y1_indicator, y2_indicator

    # tra ve chuoi user_input:
    # chuoi item_input:
    # labels:
    # y1_indicator:
    # y2_indicator:
    # cac chuoi nay co kich thuoc bang nhau.

    # get train data to fit model (can thay doi).
    # file_path: partitioned_train_data/part_000
    # interact matrix (loading from _explicit.train.rating)
    # num_negative: set from hyper parameter.
    # num_user
    # num_item.


def get_train_instances_partition_s_ite(uid, itemid, interact, interact_mat, num_negatives, num_item):
    user_input, item_input, labels, y1_indicator, y2_indicator = [], [], [], [], []
    # implicit
    user_input.append(uid)
    item_input.append(itemid)
    labels.append(1)
    y1_indicator.append(1)
    y2_indicator.append(0)

    # negative instances for implicit.
    # Lay mau cac item chua tung view hay purchase; train_data_mt = 1__purchase, -1__view
    for _ in range(num_negatives):
        j = random.randrange(num_item)  # select number from 0 - num_item.
        while (uid, j) in interact_mat:
            j = random.randrange(num_item)
        user_input.append(uid)
        item_input.append(j)
        labels.append(0)
        y1_indicator.append(1)
        y2_indicator.append(0)

    # lay mau cho explicit.
    if interact:  # neu co tuong tac explicit.
        user_input.append(uid)
        item_input.append(itemid)
        labels.append(1)
        y1_indicator.append(0)
        y2_indicator.append(1)

        for _ in range(num_negatives):
            j = random.randrange(num_item)
            while (uid, j) in interact_mat:
                j = random.randrange(num_item)
            user_input.append(uid)
            item_input.append(j)
            labels.append(0)
            y1_indicator.append(0)
            y2_indicator.append(1)
    else:  # neu khong co tuong tac explicit ( sample 5 negative cho tuong tac explicit)
        for _ in range(num_negatives + 1):
            j = random.randrange(num_item)
            while (uid, j) in interact_mat:
                j = random.randrange(num_item)
            user_input.append(uid)
            item_input.append(j)
            labels.append(0)
            y1_indicator.append(0)
            y2_indicator.append(1)

    return user_input, item_input, labels, y1_indicator, y2_indicator
    # tra ve chuoi user_input:
    # chuoi item_input:
    # labels:
    # y1_indicator:
    # y2_indicator:
    # cac chuoi nay co kich thuoc bang nhau.


def count_im_ex(file_path):
    start = datetime.datetime.now()
    # Construct matrix
    # mat = sp.dok_matrix((num_user, num_item), dtype=np.bool_)
    count_ex = 0
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            ex = int(line[5])
            if ex > 0:
                count_ex += 1
    print(count_ex)


if __name__ == '__main__':
    retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/retail_rocket/'
    recobell_root_path = settings.DATA_ROOT_PATH + "site_data/recobell/"
    ml_1m_root_path = settings.DATA_ROOT_PATH + "site_data/ml-1m/"
    # count_im_ex(retail_rocket_root_path + "scene_1/_explicit.train.rating")
    # count_im_ex(recobell_root_path + "scene_1/_explicit.train.rating")
    i = 0
    for i in range(50000000):
        i += 1
        if i % 500000 == 0:
            print "Next data", i/10000
    load_test_data_s_ite(retail_rocket_root_path + 'ratings_test.txt')
