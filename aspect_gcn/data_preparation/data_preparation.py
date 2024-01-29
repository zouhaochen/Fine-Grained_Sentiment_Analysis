# lib
import csv
import logging
import os
import random
import pandas as pd
import progressbar

random.seed(2312)


def combine_implicit_with_explicit_behaviors(implicit_input, explicit_input, user_index_output, item_index_output,
                                             output):
    """
        @param implicit_input: format raw_user_id,raw_item_id,timestamp
        @param explicit_input: format raw_user_id,raw_item_id,timestamp
        @param output: format raw_user_id,raw_item_id,implicit_timestamp,num_implicicit,explicit_timestamp,num_explicit
        @param item_index_output: file_name str
        @param user_index_output: file_name str
        """
    widgets = [" ", " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]

    # create user_id blacklist
    logging.info("Create user_id blacklist ...")
    interacted_dict = {}

    file_pointer = open(implicit_input, "r")
    csv_reader = csv.reader(file_pointer, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    for line in progressbar.ProgressBar(widgets=widgets)(csv_reader):
        raw_user_id = line[0]
        raw_item_id = line[1]
        if raw_user_id in interacted_dict:
            interacted_dict[raw_user_id].add(raw_item_id)
        else:
            interacted_dict[raw_user_id] = {raw_item_id}
    file_pointer.close()

    user_id_blacklist = set()
    for user_id in interacted_dict:
        if len(interacted_dict[user_id]) < 5:
            user_id_blacklist.add(user_id)
    print("len user_id_blacklist:", len(user_id_blacklist))

    # convert user_id, item_id
    logging.info("convert user_id, item_id")

    # user_id_dict, item_id_dict la anh xa tu raw_id sang id dang so nguyen tu 0 den n
    user_id_dict = {}
    item_id_dict = {}

    implicit_rating_dict = {}
    user_id_count = 0
    item_id_count = 0

    file_pointer = open(implicit_input, "r")
    csv_reader = csv.reader(file_pointer, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    for line in progressbar.ProgressBar(widgets=widgets)(csv_reader):
        raw_user_id = line[0]  # string
        if raw_user_id not in user_id_blacklist:
            raw_item_id = line[1]
            raw_timestamp = line[2]
            if raw_user_id not in user_id_dict:
                user_id_dict[raw_user_id] = user_id_count
                user_id_count += 1
            if raw_item_id not in item_id_dict:
                item_id_dict[raw_item_id] = item_id_count
                item_id_count += 1

            # add user data to dict
            user_id = user_id_dict[raw_user_id]  # int
            if user_id in implicit_rating_dict:
                implicit_rating_dict[user_id].append((item_id_dict[raw_item_id], raw_timestamp))
            else:
                implicit_rating_dict[user_id] = [(item_id_dict[raw_item_id], raw_timestamp)]
    file_pointer.close()

    file_pointer = open(explicit_input, "r")
    csv_reader = csv.reader(file_pointer, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    eval_users = set()
    weird_items = set()
    num_weird_ex = 0
    for line in progressbar.ProgressBar(widgets=widgets)(csv_reader):
        raw_user_id = line[0]
        if raw_user_id in user_id_dict:
            eval_users.add(raw_user_id)
            raw_item_id = line[1]
            raw_timestamp = line[2]
            if raw_item_id not in item_id_dict:
                weird_items.add(raw_item_id)

    file_pointer.close()
    print("len user list:", len(user_id_dict))
    print("len item list:", len(item_id_dict))
    print("len weird explicit user:", len(eval_users))
    print("len weird explicit item:", len(weird_items))
    print("len weird explicit:", num_weird_ex)
    print("len implicit_rating_dict:", len(implicit_rating_dict))

    with open(user_index_output, "w") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for k in user_id_dict:
            csv_writer.writerow([k, user_id_dict[k]])
    with open(item_index_output, "w") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for k in item_id_dict:
            csv_writer.writerow([k, item_id_dict[k]])
    # load explicit data put to explicit_rating_dict
    explicit_rating_dict = {}
    file_pointer = open(explicit_input, "r")
    #
    # csv_reader = csv.reader(file_pointer, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    # for line in csv_reader:
    #     raw_user_id = line[0]
    #     raw_item_id = line[1]
    #     raw_timestamp = line[2]
    #     if (raw_user_id in user_id_dict) and (raw_item_id in item_id_dict):
    #         user_id = user_id_dict[raw_user_id]
    #         if user_id in explicit_rating_dict:
    #             explicit_rating_dict[user_id].append((item_id_dict[raw_item_id], raw_timestamp))
    #         else:
    #             explicit_rating_dict[user_id] = [(item_id_dict[raw_item_id], raw_timestamp)]
    # file_pointer.close()
    #
    # # gen ratings data
    # logging.info("gen ratings with explicit data")
    # out_file = open(output, "w")
    # csv_writer = csv.writer(out_file, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    #
    # for user_id in implicit_rating_dict:
    #     # check duplicate
    #     implicit_item_time_dict = {}
    #     explicit_item_time_dict = {}
    #     for his in implicit_rating_dict[user_id]:
    #         item_id = his[0]
    #         timestamp = his[1]
    #         if item_id in implicit_item_time_dict:
    #             if timestamp > implicit_item_time_dict[item_id][0]:
    #                 implicit_item_time_dict[item_id][0] = timestamp
    #             implicit_item_time_dict[item_id][1] += 1
    #         else:
    #             implicit_item_time_dict[item_id] = [timestamp, 1]
    #
    #     if user_id in explicit_rating_dict:
    #         for his in explicit_rating_dict[user_id]:
    #             item_id = his[0]
    #             timestamp = his[1]
    #             if item_id in explicit_item_time_dict:
    #                 if timestamp > explicit_item_time_dict[item_id][0]:
    #                     explicit_item_time_dict[item_id][0] = timestamp
    #                 explicit_item_time_dict[item_id][1] += 1
    #             else:
    #                 explicit_item_time_dict[item_id] = [timestamp, 1]
    #
    #     # write to file
    #     for i in implicit_item_time_dict:
    #         if i in explicit_item_time_dict:
    #             if implicit_item_time_dict[i][0] > explicit_item_time_dict[i][0]:
    #                 implicit_item_time_dict[i][0] = explicit_item_time_dict[i][0]
    #             csv_writer.writerow([user_id, i] + implicit_item_time_dict[i] + explicit_item_time_dict[i])
    #         else:
    #             csv_writer.writerow([user_id, i] + implicit_item_time_dict[i] + [-1, 0])
    #     for j in explicit_item_time_dict:
    #         if j not in implicit_item_time_dict:
    #             csv_writer.writerow([user_id, j] + explicit_item_time_dict[j] + explicit_item_time_dict[j])
    # out_file.close()


def div_train_test_data_without_implicit_in_train(input_file, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Local function ---------------------------------------------------------------
    def process_and_write_record(rate_list_of_a_user: list):
        test_record = []
        num_negative = 999

        # # filter items being explicit interact
        # ex_item_list = [int(i[1]) for i in rate_list_of_a_user if int(i[4]) > 0]
        #
        # # negative item set for test
        # total_item_list = range(num_items)
        # neg_item_list = list(set(total_item_list) - set(ex_item_list))
        # rand_neg_item_list = random.sample(neg_item_list, num_negative)

        # select test record
        ex_max_time = 0
        ex_max_idx = -1
        for idx, record in enumerate(rate_list_of_a_user):
            ex_time = int(record[4])
            if ex_time >= ex_max_time:
                ex_max_time = ex_time
                test_record = record
                ex_max_idx = idx

        # create test and negative-test
        if ex_max_time > 0:
            # csv_writer_explicit_test.writerow(test_record[0:2] + test_record[4:])
            # explicit_negative_record_line = ["(" + str(test_record[0]) + "," + str(test_record[1]) + ")"]
            # explicit_negative_record_line += rand_neg_item_list
            # csv_writer_explicit_negative.writerow(explicit_negative_record_line)

            rate_list_of_a_user.remove(test_record)  # warning: list.remove() only remove the first occurrence of item

        # create train
        for record in rate_list_of_a_user:
            csv_writer_train.writerow(record)

    # End function ---------------------------------------------------------------

    # get num_users and num_items
    logging.info("Get num_users and num_items")
    num_users = 0
    num_items = 0
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            num_users = max(num_users, int(line[0]))
            num_items = max(num_items, int(line[1]))
    num_users += 1
    num_items += 1
    logging.info("num users: " + str(num_users) + ";    num items: " + str(num_items))

    # gen train, test, negative data
    logging.info("gen train, test, negative data")

    train_file = open(output_dir + "_explicit.train.rating", "w")
    csv_writer_train = csv.writer(train_file, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    explicit_test_file = open(output_dir + "_explicit.test.rating", "w")
    csv_writer_explicit_test = csv.writer(explicit_test_file, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    explicit_negative_file = open(output_dir + "_explicit.test.negative", "w")
    csv_writer_explicit_negative = csv.writer(explicit_negative_file, delimiter="|", quotechar="",
                                              quoting=csv.QUOTE_NONE)

    tmp_user_id = 0
    item_list_of_tmp_user = []
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            # type line = [user_id item_id time_im num_im time_ex num_ex]
            user_id = int(line[0])
            if user_id != tmp_user_id:
                process_and_write_record(item_list_of_tmp_user)
                tmp_user_id = user_id
                item_list_of_tmp_user = []
            item_list_of_tmp_user.append(line)
        # end for    
        process_and_write_record(item_list_of_tmp_user)

    train_file.close()
    explicit_test_file.close()
    explicit_negative_file.close()


def div_train_test_data_with_implicit_in_train(input_file, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Local function ---------------------------------------------------------------
    def process_and_write_record(rate_list_of_a_user: list):
        test_record = []
        num_negative = 999

        # filter items being explicit interact
        ex_item_list = [int(i[1]) for i in rate_list_of_a_user if int(i[4]) > 0]

        # negative item set for test
        total_item_list = range(num_items)
        neg_item_list = list(set(total_item_list) - set(ex_item_list))
        rand_neg_item_list = random.sample(neg_item_list, num_negative)

        # select test record
        ex_max_time = 0
        ex_max_idx = -1
        for idx, record in enumerate(rate_list_of_a_user):
            ex_time = int(record[4])
            if ex_time >= ex_max_time:
                ex_max_time = ex_time
                test_record = record
                ex_max_idx = idx

        # create test and negative-test
        if ex_max_time > 0:
            csv_writer_explicit_test.writerow(test_record[0:2] + test_record[4:])
            explicit_negative_record_line = ["(" + str(test_record[0]) + "," + str(test_record[1]) + ")"]
            explicit_negative_record_line += rand_neg_item_list
            csv_writer_explicit_negative.writerow(explicit_negative_record_line)

            # rate_list_of_a_user.remove(test_record)  # warning: list.remove() only remove the first occurrence of item
            rate_list_of_a_user[ex_max_idx][4] = -1
            rate_list_of_a_user[ex_max_idx][5] = 0

        # create train
        for record in rate_list_of_a_user:
            csv_writer_train.writerow(record)

    # End function ---------------------------------------------------------------

    # get num_users and num_items
    logging.info("Get num_users and num_items")
    num_users = 0
    num_items = 0
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            num_users = max(num_users, int(line[0]))
            num_items = max(num_items, int(line[1]))
    num_users += 1
    num_items += 1
    logging.info("num users: " + str(num_users) + ";    num items: " + str(num_items))

    # gen train, test, negative data
    logging.info("gen train, test, negative data")

    train_file = open(output_dir + "_explicit.train.rating", "w")
    csv_writer_train = csv.writer(train_file, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    explicit_test_file = open(output_dir + "_explicit.test.rating", "w")
    csv_writer_explicit_test = csv.writer(explicit_test_file, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    explicit_negative_file = open(output_dir + "_explicit.test.negative", "w")
    csv_writer_explicit_negative = csv.writer(explicit_negative_file, delimiter="|", quotechar="",
                                              quoting=csv.QUOTE_NONE)

    tmp_user_id = 0
    item_list_of_tmp_user = []
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            # type line = [user_id item_id time_im num_im time_ex num_ex]
            user_id = int(line[0])
            if user_id != tmp_user_id:
                process_and_write_record(item_list_of_tmp_user)
                tmp_user_id = user_id
                item_list_of_tmp_user = []
            item_list_of_tmp_user.append(line)
        # end for
        process_and_write_record(item_list_of_tmp_user)

    train_file.close()
    explicit_test_file.close()
    explicit_negative_file.close()


def filter_active_user_and_item(implicit_input, explicit_input, user_index_output, item_index_output, implicit_output,
                                explicit_output):
    widgets = [" ", " ", progressbar.SimpleProgress(), " ", progressbar.Timer()]

    # create user_id blacklist
    logging.info("Creating user_id blacklist ...")
    interacted_dict = {}

    implicit_input_fp = open(implicit_input, "r")
    implicit_input_csv_reader = csv.reader(implicit_input_fp, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    for line in progressbar.ProgressBar(widgets=widgets)(implicit_input_csv_reader):
        raw_user_id = line[0]
        raw_item_id = line[1]
        if raw_user_id in interacted_dict:
            interacted_dict[raw_user_id].add(raw_item_id)
        else:
            interacted_dict[raw_user_id] = {raw_item_id}
    implicit_input_fp.close()

    user_id_blacklist = set()
    for user_id in interacted_dict:
        if len(interacted_dict[user_id]) < 5:
            user_id_blacklist.add(user_id)
    print("len user_id_blacklist:", len(user_id_blacklist))

    # convert user_id, item_id
    logging.info("convert user_id, item_id")

    # user_id_dict, item_id_dict la anh xa tu raw_id sang id dang so nguyen tu 0 den n
    user_id_dict = {}
    item_id_dict = {}

    # implicit_rating_dict = {}
    user_id_count = 0
    item_id_count = 0

    implicit_input_fp = open(implicit_input, "r")
    implicit_output_fp = open(implicit_output, "w")
    explicit_input_fp = open(explicit_input, "r")
    explicit_output_fp = open(explicit_output, "w")
    implicit_input_csv_reader = csv.reader(implicit_input_fp, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    implicit_output_csv_writer = csv.writer(implicit_output_fp, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    explicit_input_csv_reader = csv.reader(explicit_input_fp, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    explicit_output_csv_writer = csv.writer(explicit_output_fp, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)

    eval_users = set()
    weird_items = set()
    num_eval = 0
    num_training = 0

    for line in progressbar.ProgressBar(widgets=widgets)(implicit_input_csv_reader):
        raw_user_id = line[0]
        if raw_user_id not in user_id_blacklist:
            num_training += 1
            raw_item_id = line[1]
            # raw_timestamp = line[2]
            if raw_user_id not in user_id_dict:
                user_id_dict[raw_user_id] = user_id_count
                user_id_count += 1
            if raw_item_id not in item_id_dict:
                item_id_dict[raw_item_id] = item_id_count
                item_id_count += 1
            line[0] = user_id_dict[raw_user_id]
            line[1] = item_id_dict[raw_item_id]
            implicit_output_csv_writer.writerow(line)

    implicit_input_fp.close()
    implicit_output_fp.close()

    for line in progressbar.ProgressBar(widgets=widgets)(explicit_input_csv_reader):
        raw_user_id = line[0]
        if raw_user_id in user_id_dict:
            num_eval += 1
            eval_users.add(raw_user_id)
            raw_item_id = line[1]
            if raw_item_id not in item_id_dict:
                weird_items.add(raw_item_id)
                item_id_dict[raw_item_id] = item_id_count
                item_id_count += 1

            line[0] = user_id_dict[raw_user_id]
            line[1] = item_id_dict[raw_item_id]
            explicit_output_csv_writer.writerow(line)
    explicit_input_fp.close()
    explicit_output_fp.close()
    print("len user list:", len(user_id_dict))
    print("len item list:", len(item_id_dict))
    print("num eval user:", len(eval_users))
    print("num weird item:", len(weird_items))
    print("num training:", num_training)
    print("num eval:", num_eval)

    with open(user_index_output, "w") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for k in user_id_dict:
            csv_writer.writerow((k, user_id_dict[k]))
    with open(item_index_output, "w") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for k in item_id_dict:
            csv_writer.writerow((k, item_id_dict[k]))


def combine_implicit_with_explicit_data(implicit_input, explicit_input, output):
    implicit_pd_df = pd.read_csv(implicit_input, header=None, names=["uid", "itemid", "time", "label"])
    explicit_pd_df = pd.read_csv(explicit_input, header=None, names=["uid", "itemid", "time", "label"])
    combined_pd_df = implicit_pd_df \
        .append(explicit_pd_df) \
        .sort_values(by=["uid", "time", "itemid"]) \
        .reset_index(drop=True)
    combined_pd_df.to_csv(output, header=False, index=False)


def new_div_train_test_data_without_implicit_in_train(input_file, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Local function ---------------------------------------------------------------
    def process_and_write():
        interacted_item_set = set()
        num_negative = 50

        # find test record
        test_item_id = -1
        test_idx = -1
        for idx, record in enumerate(behavior_list):
            item_id = int(record[1])
            label = int(record[3])

            interacted_item_set.add(item_id)
            if label > 1:
                test_item_id = item_id
                test_idx = idx

        if test_idx >= 0:
            # create test and negative-test
            test_csv_writer.writerow(behavior_list[test_idx])
            try:
                rand_neg_test_list = random.sample(set_of_all_item - interacted_item_set, k=num_negative)
            except ValueError:
                rand_neg_test_list = random.choices(set_of_all_item - interacted_item_set, k=num_negative)
            test_and_neg = behavior_list[test_idx][0:2] + rand_neg_test_list
            neg_test_csv_writer.writerow(test_and_neg)

            # create train
            for record in behavior_list:
                if int(record[1]) != test_item_id:
                    train_csv_writer.writerow(record)
        else:
            for record in behavior_list:
                train_csv_writer.writerow(record)
        # End function ---------------------------------------------------------------

    # get num_user and num_item
    logging.info("Get num_user and num_item")
    num_user = 0
    num_item = 0
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            num_user = max(num_user, int(line[0]))
            num_item = max(num_item, int(line[1]))
    num_user += 1
    num_item += 1
    logging.info("num user: " + str(num_user) + ";    num item: " + str(num_item))

    # gen train, test, negative data
    logging.info("gen train, test, negative data")

    train_fp = open(output_dir + "_train.rating", "w")
    train_csv_writer = csv.writer(train_fp, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    test_fp = open(output_dir + "_test.rating", "w")
    test_csv_writer = csv.writer(test_fp, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    neg_test_fp = open(output_dir + "_test.negative", "w")
    neg_test_csv_writer = csv.writer(neg_test_fp, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    tmp_user_id = 0
    behavior_list = []
    set_of_all_item = set(range(num_item))
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            # type line = [user_id item_id time_im num_im time_ex num_ex]
            user_id = int(line[0])
            if user_id != tmp_user_id:
                process_and_write()
                tmp_user_id = user_id
                behavior_list = []
            behavior_list.append(line)
        # end for
        process_and_write()

    train_fp.close()
    test_fp.close()
    neg_test_fp.close()


def new_div_train_test_data_with_implicit_in_train(input_file, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Local function ---------------------------------------------------------------
    def process_and_write():
        interacted_item_set = set()
        num_negative = 999

        # find test record
        test_item_id = -1
        test_idx = -1
        for idx, record in enumerate(behavior_list):
            item_id = int(record[1])
            label = int(record[3])

            interacted_item_set.add(item_id)
            if label > 1:
                test_item_id = item_id
                test_idx = idx

        if test_idx >= 0:
            # create test and negative-test
            test_csv_writer.writerow(behavior_list[test_idx])
            try:
                rand_neg_test_list = random.sample(set_of_all_item - interacted_item_set, k=num_negative)
            except ValueError:
                rand_neg_test_list = random.choices(set_of_all_item - interacted_item_set, k=num_negative)
            test_and_neg = behavior_list[test_idx][0:2] + rand_neg_test_list
            neg_test_csv_writer.writerow(test_and_neg)

            # create train
            for idx, record in enumerate(behavior_list):
                if int(record[1]) != test_item_id:
                    train_csv_writer.writerow(record)
                elif idx != test_idx:
                    record[3] = 1
                    train_csv_writer.writerow(record)
        else:
            for record in behavior_list:
                train_csv_writer.writerow(record)
        # End function ---------------------------------------------------------------

    # get num_user and num_item
    logging.info("Get num_user and num_item")
    num_user = 0
    num_item = 0
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in csv_reader:
            num_user = max(num_user, int(line[0]))
            num_item = max(num_item, int(line[1]))
    num_user += 1
    num_item += 1
    logging.info("num user: " + str(num_user) + ";    num item: " + str(num_item))

    # gen train, test, negative data
    logging.info("gen train, test, negative data")

    train_fp = open(output_dir + "_train.rating", "w")
    train_csv_writer = csv.writer(train_fp, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    test_fp = open(output_dir + "_test.rating", "w")
    test_csv_writer = csv.writer(test_fp, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    neg_test_fp = open(output_dir + "_test.negative", "w")
    neg_test_csv_writer = csv.writer(neg_test_fp, delimiter="|", quotechar="", quoting=csv.QUOTE_NONE)

    tmp_user_id = 0
    behavior_list = []
    set_of_all_item = set(range(num_item))
    with open(input_file, "r") as f:
        csv_reader = csv.reader(f, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
        for line in progressbar.ProgressBar()(csv_reader):
            # type line = [user_id item_id time_im num_im time_ex num_ex]
            user_id = int(line[0])
            if user_id != tmp_user_id:
                process_and_write()
                tmp_user_id = user_id
                behavior_list = []
            behavior_list.append(line)
        # end for
        process_and_write()

    train_fp.close()
    test_fp.close()
    neg_test_fp.close()
