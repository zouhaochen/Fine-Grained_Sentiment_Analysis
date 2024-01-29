"""
Tien xu ly du lieu cua recobell data
"""
import csv
import datetime
import logging
import subprocess
import time

import pandas as pd
import progressbar

from src.config import config

from src.data_preparation import data_preparation

# config.py log
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

data_name = "recobell/"
root_path = config.DATA_ROOT_PATH + data_name


def convert_time(d):
    try:
        new_d = int(time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f").timetuple()))
    except ValueError:
        new_d = int(time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple()))
    return new_d


def get_new_tiny_site_data(input_file, output_file):
    start_date = convert_time("2016-08-01 00:00:00.000")
    finish_date = convert_time("2016-09-01 00:00:00.000")

    input_fp = open(input_file, "r")
    output_fp = open(output_file, "w")

    csv_reader = csv.reader(input_fp, delimiter=",", quotechar="\"", quoting=csv.QUOTE_ALL)
    csv_writer = csv.writer(output_fp, delimiter=",", quotechar="\"", quoting=csv.QUOTE_ALL)

    for line in progressbar.ProgressBar()(csv_reader):
        raw_timestamp = line[0]
        timestamp = convert_time(raw_timestamp)
        if start_date <= timestamp < finish_date:
            csv_writer.writerow(line)
    input_fp.close()
    output_fp.close()


def gen_implicit_cleaned_data(input_file, output_file):
    start_date = convert_time("2016-08-08 00:00:00.000")
    finish_date = convert_time("2016-08-15 00:00:00.000")

    input_fp = open(input_file, "r")
    output_fp = open(output_file, "w")
    csv_reader = csv.reader(input_fp, delimiter=",", quotechar="\"", quoting=csv.QUOTE_ALL)
    csv_writer = csv.writer(output_fp, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    # csv_writer_2 = csv.writer(f_2, delimiter=",", quotechar=""", quoting=csv.QUOTE_ALL)

    for line in progressbar.ProgressBar()(csv_reader):
        raw_timestamp = line[0]
        raw_uid = line[3]
        raw_item_id = line[4]
        timestamp = convert_time(raw_timestamp)

        if start_date <= timestamp < finish_date:
            csv_writer.writerow((raw_uid, raw_item_id, timestamp, 1))
    input_fp.close()
    output_fp.close()


def gen_explicit_cleaned_data(input_file, output_file):
    logging.info("gen_explicit_cleaned_data")

    start_date = convert_time("2016-08-08 00:00:00.000")
    finish_date = convert_time("2016-08-15 00:00:00.000")
    # convert uid, item_id

    file_out = open(output_file, "w")
    csv_writer = csv.writer(file_out, delimiter=",", quotechar="", quoting=csv.QUOTE_NONE)
    with open(input_file, "r") as f:

        csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_ALL)

        for line in progressbar.ProgressBar()(csv_reader):
            raw_timestamp = line[0]
            raw_uid = line[3]
            raw_item_id = line[4]
            timestamp = convert_time(raw_timestamp)

            if start_date <= timestamp < finish_date:
                csv_writer.writerow([raw_uid, raw_item_id, timestamp, 2])
    file_out.close()


def construct_item_vector():
    item_recobell_df = pd.read_csv(root_path + "raw_data/site_product.csv000",
                                   header=None,
                                   names=["itemid", "price", "cat1", "cat2", "cat3", "cat4", "brandid"])

    cat_1_list = list(set(item_recobell_df.cat1))
    cat_1_dict = {cat_1_list[i]: i for i in range(len(cat_1_list))}
    a = len(cat_1_list)
    cat_2_list = list(set(item_recobell_df.cat2))
    cat_2_dict = {cat_2_list[i]: i + a for i in range(len(cat_2_list))}
    a += len(cat_2_list)
    cat_3_list = list(set(item_recobell_df.cat3))
    cat_3_dict = {cat_3_list[i]: i + a for i in range(len(cat_3_list))}
    a += len(cat_3_list)
    cat_4_list = list(set(item_recobell_df.cat4))
    cat_4_dict = {cat_4_list[i]: i + a for i in range(len(cat_4_list))}
    a += len(cat_4_list)
    with open("../../data/recobell_cat1_index.txt", "w") as f:
        for k in cat_1_dict:
            f.write(k + "," + str(cat_1_dict[k]) + "\n")
    with open("../../data/recobell_cat2_index.txt", "w") as f:
        for k in cat_2_dict:
            f.write(k + "," + str(cat_2_dict[k]) + "\n")
    with open("../../data/recobell_cat3_index.txt", "w") as f:
        for k in cat_3_dict:
            f.write(k + "," + str(cat_3_dict[k]) + "\n")
    with open("../../data/recobell_cat4_index.txt", "w") as f:
        for k in cat_4_dict:
            f.write(k + "," + str(cat_4_dict[k]) + "\n")
    # cat_1_dict = {}
    # cat_2_dict = {}
    # cat_3_dict = {}
    # cat_4_dict = {}
    # with open(root_path + "recobell_cat1_index.txt", "r") as f:
    #     for line in f:
    #         s_line = line.split(",")
    #         cat_1_dict[s_line[0]] = int(s_line[1])
    # print(cat_1_dict)
    # with open(root_path + "recobell_cat2_index.txt", "r") as f:
    #     for line in f:
    #         s_line = line.split(",")
    #         cat_2_dict[s_line[0]] = int(s_line[1])
    # with open(root_path + "recobell_cat3_index.txt", "r") as f:
    #     for line in f:
    #         s_line = line.split(",")
    #         cat_3_dict[s_line[0]] = int(s_line[1])
    # with open(root_path + "recobell_cat4_index.txt", "r") as f:
    #     for line in f:
    #         s_line = line.split(",")
    #         cat_4_dict[s_line[0]] = int(s_line[1])
    # with open(root_path + "item_repr.txt", "w") as f:
    #     csv_writer = csv.writer(f, delimiter=",", quotechar="|", quoting=csv.QUOTE_ALL)
    #     for item in item_recobell_df.iterrows():
    #         sp_vec = {cat_1_dict[item[1].cat1]: 1.0, cat_2_dict[item[1].cat2]: 1.0, cat_3_dict[item[1].cat3]: 1.0,
    #                   cat_4_dict[item[1].cat4]: 1.0}
    #         sp_repr = dict_sparse_vector_to_json_string(sp_vec)
    #         csv_writer.writerow((item[1].itemid, sp_repr))
    # print("Done infer pcat repr of recobell items !")


def main():
    raw_implicit_data_path = root_path + "raw_data/site_view_log.csv000"
    raw_explicit_data_path = root_path + "raw_data/site_order_log.csv000"
    tiny_implicit_data_path = root_path + "raw_data/tiny_site_view_log.csv000"
    tiny_explicit_data_path = root_path + "raw_data/tiny_site_order_log.csv000"
    cleaned_implicit_data_path = root_path + "_implicit.clean"
    cleaned_explicit_data_path = root_path + "_explicit.clean"
    filtered_implicit_data_path = root_path + "_implicit.filter"
    filtered_explicit_data_path = root_path + "_explicit.filter"
    sorted_filtered_combined_data_path = root_path + "_combine.filter.sorted"

    user_index_dict = root_path + "u2index.txt"
    item_index_dict = root_path + "i2index.txt"
    output_dir = root_path + 'without_implicit_in_train/'
    # output_dir = root_path + 'with_implicit_in_train/'

    # # implicit
    # logging.info("Getting new tiny implicit site data ...")
    # get_new_tiny_site_data(raw_implicit_data_path, tiny_implicit_data_path)
    # logging.info("Cleaning tiny implicit data ...")
    # gen_implicit_cleaned_data(tiny_implicit_data_path, cleaned_implicit_data_path)
    # logging.info("--> Done !!!")
    #
    # # explicit
    # logging.info("Getting new tiny explicit site data ...")
    # get_new_tiny_site_data(raw_explicit_data_path, tiny_explicit_data_path)
    # logging.info("Cleaning tiny implicit data ...")
    # gen_explicit_cleaned_data(tiny_explicit_data_path, cleaned_explicit_data_path)
    # logging.info("--> Done !!!")

    # # filter active user and item in implicit and explicit data
    # logging.info("Filtering active user and item in implicit and explicit data")
    # data_preparation.filter_active_user_and_item(cleaned_implicit_data_path, cleaned_explicit_data_path,
    #                                              user_index_dict, item_index_dict, filtered_implicit_data_path,
    #                                              filtered_explicit_data_path)
    # logging.info("--> Done !!!")

    # # Combine implicit and explicit data and sort
    # logging.info("Combine implicit and explicit data, then sort by time")
    # data_preparation.combine_implicit_with_explicit_data(filtered_implicit_data_path, filtered_explicit_data_path,
    #                                                      sorted_filtered_combined_data_path)
    # logging.info("--> Done !!!")

    # # Div to train, test without implicit in train
    # data_preparation.new_div_train_test_data_without_implicit_in_train(sorted_filtered_combined_data_path, output_dir)
    # logging.info("--> Done !!!")

    logging.info("--> Split train data into partition")
    subprocess.call(["bash", "src/data_preparation/bin/split.sh", output_dir])

    # data_for_VALS.preprocessing_for_VALS(data_name)
    # logging.info("--> Done, for_VALS")

    # data_preparation.new_div_train_test_data_with_implicit_in_train(sorted_filtered_combined_data_path, output_dir)


if __name__ == "__main__":
    main()
