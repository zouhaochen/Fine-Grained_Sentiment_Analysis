import progressbar
from src import settings
import pandas as pd
import csv
import random

root_path = settings.DATA_ROOT_PATH + 'site_data/tmall_dataset/'
data_file = root_path + "1101_1130.csv"


def get_split_data_file():  # generate text file.
    print("reading...")
    ratings = pd.read_csv(data_file, delimiter=',', header=None, engine='python',
                          names=['use_ID', 'sel_ID', 'ite_ID', 'cat_ID', 'act_ID', 'time'])
    print("Head:")
    print(ratings[:5])
    # xoa di mot so cot

    print("Drop 1 column")
    ratings = ratings.drop(ratings.columns[[1, 3]], axis=1)
    print(ratings[:5])

    print("Group and ordering...")
    ratings = ratings.groupby(["use_ID"]).apply(
        lambda x: x.sort_values(["use_ID", "time"], ascending=True)).reset_index(drop=True)

    # lay nhung user co so luong tuong tac > 5
    print("Length before filter:", len(ratings))
    ratings = ratings.groupby('use_ID').filter(lambda x: len(x) > 5)

    # ghi ra file text co thu tu.
    print("Printing...")
    out_file = root_path + "1101_1130_clean_filter.csv"
    ratings.to_csv(out_file, header=None, index=None, sep=',')


# get_split_data_file()


def preprocess_text_file():  # create u2index and i2index
    data_file = root_path + "1101_1130_clean_filter.csv"  # input
    user_index_dict = root_path + 'u2index.txt'
    item_index_dict = root_path + 'i2index.txt'

    # convert old id user, item into new id.
    # uid_dict, item_id_dict la anh xa tu raw_id sang id dang so nguyen tu 0 den n
    user_id_dict = {}
    item_id_dict = {}

    rating_dict = {}
    uid_count = 0
    item_id_count = 0

    # tiep tuc tu implicit input
    file_pointer2 = open(data_file, "r")
    csv_reader2 = csv.reader(file_pointer2, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    # 6, 1553354, 0, 20151125
    # 6, 1553354, 0, 20151125
    # 6, 459143, 0, 20151125
    # 6, 365915, 0, 20151125
    # 6, 1497816, 1, 20151125
    # 6, 1497816, 0, 20151125

    for line in progressbar.ProgressBar()(csv_reader2):  # moi dong trong implicit data.
        raw_uid = line[0]  # string user_id
        raw_item_id = line[1]  # item id
        interact = line[2]
        timestamp = line[3]  # time click
        if raw_uid not in user_id_dict:  # tao ra dict user_id_dict  raw_user_id => stt cua user do.
            user_id_dict[raw_uid] = uid_count
            uid_count += 1
        if raw_item_id not in item_id_dict:  # dict item: key la item_id => so tt cua item do
            item_id_dict[raw_item_id] = item_id_count
            item_id_count += 1

        # interact new uid tuong ung voi raw_uid
        uid = user_id_dict[raw_uid]
        # int : lay ra uid moi cua user tuong ung voi raw_uid
        if uid in rating_dict:
            rating_dict[uid].append((item_id_dict[raw_item_id], interact, timestamp))
        else:
            rating_dict[uid] = [(item_id_dict[raw_item_id], interact, timestamp)]
        # rating_dict[new_uid]: value la mot mang voi moi phan tu la new_item_id, interact, raw_time_stamp
    file_pointer2.close()
    print("len user list:", len(user_id_dict))
    print("len item list:", len(item_id_dict))
    print("len implicit_rating_dict:", len(rating_dict))

    with open(user_index_dict, 'w') as f:  # viet ra file 'u2index.txt'
        csv_writer = csv.writer(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for k in user_id_dict:
            csv_writer.writerow([k, user_id_dict[k]])  # uid ban dau => uid moi.

    with open(item_index_dict, 'w') as f:  # viet ra file 'i2index.txt'
        csv_writer = csv.writer(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for k in item_id_dict:
            csv_writer.writerow([k, item_id_dict[k]])  # item id raw => itemid moi.


# preprocess_text_file()

def replace_with_newid():
    # input
    data_file = root_path + "1101_1130_clean_filter.csv"
    user_index_dict = root_path + 'u2index.txt'
    item_index_dict = root_path + 'i2index.txt'
    # output : get 1101_1130_clean_filter.csv with new id for users, items
    output_file = root_path + '1101_1130_clean_filter_newids.txt'

    users_dict = {}
    item_dict = {}
    users = open(user_index_dict, "r")
    print("User process")
    users_reader = csv.reader(users, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in users_reader:
        raw_uid = int(line[0])
        new_uid = int(line[1])
        users_dict[raw_uid] = new_uid

    items = open(item_index_dict, "r")
    print("Item process")
    items_reader = csv.reader(items, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in items_reader:
        raw_iid = int(line[0])
        new_iid = int(line[1])
        item_dict[raw_iid] = new_iid

    data = open(data_file, "r")
    new_data = []
    data_reader = csv.reader(data, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in progressbar.ProgressBar()(data_reader):
        raw_uid = int(line[0])
        raw_iid = int(line[1])
        interact = int(line[2])
        time = int(line[3])
        new_data.append([users_dict[raw_uid], item_dict[raw_iid], interact, time])

    # write
    print("Printing....")
    with open(output_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for k in new_data:
            csv_writer.writerow([k[0], k[1], k[2], k[3]])  # item id raw => itemid moi.
    print("DOne...")


# replace_with_newid()

def create_test_negative_data():
    print("Create test negative explicit data")
    # input
    data_file = root_path + "1101_1130_clean_filter_newids.txt"

    # Local function ---------------------------------------------------------------
    def process_and_write_record_with_explicit(rate_list, num_items):
        explicit_remove = []
        num_negative = 999

        # positive record
        explicit_positive_records = [int(i[1]) for i in rate_list if int(i[2]) == 1]
        # print("Explicit positive records")
        # print(explicit_positive_records)
        # print(set(explicit_positive_records))

        # negative record only for test
        set_total_items = range(num_items)
        negative = list(set(set_total_items) - set(explicit_positive_records))
        explicit_negative_records = random.sample(negative, num_negative)

        # test record
        explicit_max_time = 0
        for i in range(len(rate_list)):
            if int(rate_list[i][2]) == 1:
                explicit_max_time = 1
                explicit_remove = rate_list[i]
        # print("Explicit remove:")
        # print(explicit_remove)

        # writing
        if explicit_max_time == 1:
            explicit_negative_record_line = ['(' + str(explicit_remove[0]) + ',' + str(explicit_remove[1]) + ')']
            explicit_negative_record_line += explicit_negative_records
            csv_writer_explicit_negative.writerow(explicit_negative_record_line)

    # get num_users and num_items
    print('Get num_users and num_items')
    num_users = 0
    num_items = 0
    with open(data_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            num_users = max(num_users, int(line[0]))
            num_items = max(num_items, int(line[1]))
    num_users += 1
    num_items += 1

    # output file
    explicit_negative_file = open(root_path + '1101_1130_negative.txt', 'w')
    csv_writer_explicit_negative = csv.writer(explicit_negative_file, delimiter='|', quotechar='',
                                              quoting=csv.QUOTE_NONE)
    tmp_uid = 0
    tmp_item_list_single_user = []
    with open(data_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            # type line = [uid :item_id :interact :time]
            uid = int(line[0])
            if uid != tmp_uid:
                process_and_write_record_with_explicit(tmp_item_list_single_user, num_items)
                tmp_uid = uid
                tmp_item_list_single_user = []
            tmp_item_list_single_user.append(line)
        # end for
        process_and_write_record_with_explicit(tmp_item_list_single_user, num_items)


# create_test_negative_data()


def create_train_test():  # split data into train, test.
    print("Create train test....")
    retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/tmall_dataset/'
    data_file = open(retail_rocket_root_path + "1101_1130_clean_filter_newids.txt", "r")

    final_sequence = retail_rocket_root_path + "demo_1101_1130_train.txt"

    interact_item_sequences = {}  # cac chuoi item sequences tuong ung cho moi user
    interact_sequences = {}  # dict user_id va sequence interact tuong ung.
    test_explicit = {}
    new_item_sequences = {}
    new_iteract_sequences = {}
    # thong tin tuong ung voi cac sp:
    # 0: co tuong tac implicit
    # 1: co tuong tac explicit

    data_reader = csv.reader(data_file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    # for line in progressbar.ProgressBar()(data_reader):
    k = 0
    for line in data_reader:
        k += 1
        if k % 1000000 == 0:
            print "Line: ", k
        uid = int(line[0])
        iid = int(line[1])
        interact = int(line[2])

        # add into sequence item and sequence interact.
        if uid in interact_item_sequences:
            interact_item_sequences[uid].append(iid)
        else:
            interact_item_sequences[uid] = [iid]
        # add into sequence interact
        if uid in interact_sequences:
            interact_sequences[uid].append(interact)
        else:
            interact_sequences[uid] = [interact]

    # get test explicit feedback
    print("Get test explicit")
    for uid in interact_item_sequences:
        k = 0
        for interact in interact_sequences[uid]:
            if interact == 1:
                test_explicit[uid] = interact_item_sequences[uid][k]  # uid = iid.
            k += 1

    print("Len of test explicit: ", len(test_explicit.keys()))
    # remove test item from train sequence
    # get list position to remove
    print("Remove from train data")
    # neu co nhieu hon 1 phan tu test item trong tap train
    for uid in test_explicit:
        if len(interact_sequences[uid]) != len(interact_item_sequences[uid]):
            print("Not equal: ", uid)
        new_item_sequences[uid] = []
        new_iteract_sequences[uid] = []

        for k in range(min(len(interact_item_sequences[uid]),
                           len(interact_sequences[uid]))):  # moi item trong ds item tuong tac cua uid
            if interact_item_sequences[uid][k] != test_explicit[uid] or interact_sequences[uid][k] != 1:
                new_item_sequences[uid].append(interact_item_sequences[uid][k])
                new_iteract_sequences[uid].append(interact_sequences[uid][k])
    print("Init writing for training")
    writefile = open(final_sequence, "w")
    csv_writer = csv.writer(writefile, delimiter='|', quotechar='"', quoting=csv.QUOTE_NONE)
    for uid in new_item_sequences:
        csv_writer.writerow([uid, new_item_sequences[uid], new_iteract_sequences[uid]])

    print("Init writing for testing")
    test_file = open(root_path + 'demo_1101_1130_test.txt', "w")
    test_writer = csv.writer(test_file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for uid in test_explicit:
        test_writer.writerow([uid, test_explicit[uid]])
    print("Done")


create_train_test()
