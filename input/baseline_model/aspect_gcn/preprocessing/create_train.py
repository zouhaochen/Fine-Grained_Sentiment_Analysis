# writing into train file.
import csv
from src import settings

data_path = settings.DATA_ROOT_PATH + 'site_data/retail_rocket/z_data_processing/'


def preprocess():
    file_data = open(data_path + "sorted_filtered_data.txt", "r")
    remove_time = data_path + "sorted_filtered_data_remove_time.txt"
    list_items = {}
    list_interacts = {}
    for line in file_data.readlines():
        line = line.strip()
        elements = line.split(",")
        uid = int(elements[0])
        itemid = int(elements[1])
        interact = int(elements[3])

        if interact == 1:
            # list items
            if uid in list_items:
                list_items[uid].append(itemid)
            else:
                list_items[uid] = [itemid]
            # list interacts.
            if uid in list_interacts:
                list_interacts[uid].append(0)
            else:
                list_interacts[uid] = [0]
        else:
            # list items
            if uid in list_items:
                list_items[uid].append(itemid)
            else:
                list_items[uid] = [itemid]
            # list interacts.
            if uid in list_interacts:
                list_interacts[uid].append(1)
            else:
                list_interacts[uid] = [1]

    # writing to ratings.txt file (user| danh sach item| danh sach cac tuong tac)
    # tu file nay co the tao ra file training_without_implicit (xoa tat ca tuong tac cua test item)
    # tao ra file with_implicit (chuyen het cac tuong tac implicit cua test item ve 0)
    print("Init writing file")
    writefile = open(remove_time, "w")
    csv_writer = csv.writer(writefile, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
    for uid in list_items:
        csv_writer.writerow([uid, list_items[uid], list_interacts[uid]])
    print "Done"


def create_train_file_with_implicit_interact():
    file_test = open(data_path + "ratings_test_v7.txt", "r")
    list_test = {}
    for line in file_test.readlines():
        line = line.strip()
        elements = line.split(",")
        uid = int(elements[0])
        itemid = int(elements[1])
        list_test[uid] = itemid

    dict_items = {}
    dict_iteracts = {}
    train_file = open(data_path + "sorted_filtered_data_remove_time.txt")
    line_number = 0
    for line in train_file:
        line_number += 1
        if line_number % 10000 == 0:
            print "Line number: ", line_number
        list_items = []
        line = line.strip()
        elements = line.split("|")
        uid = int(elements[0])
        # list items
        itemid_string = elements[1].strip()[1:-1]
        itemid_string = itemid_string.split(",")
        for i in itemid_string:
            list_items.append(int(i.strip()))
        dict_items[uid] = list_items

        # list interacts
        list_interact = []
        interact_string = elements[2].strip()[1:-1]
        interact_string = interact_string.split(",")
        for i in interact_string:
            list_interact.append(int(i.strip()))
        dict_iteracts[uid] = list_interact

    for uid in list_test:
        for k in range(len(dict_items[uid])):
            if dict_items[uid][k] == list_test[uid]:
                dict_iteracts[uid][k] = 0

    # writing into new file train & test
    print("Init writing file")
    file_train = data_path + 'ratings_train_v7.txt'
    write_file = open(file_train, "w")
    writer = csv.writer(write_file, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
    for uid in dict_items:
        writer.writerow([uid, dict_items[uid], dict_iteracts[uid]])
    print "Done"


def create_train_file_without_implicit_interact():
    file_test = open(data_path + "ratings_test.txt", "r")
    list_test = {}
    for line in file_test.readlines():
        line = line.strip()
        elements = line.split(",")
        uid = int(elements[0])
        itemid = int(elements[1])
        list_test[uid] = itemid

    dict_items = {}
    dict_iteracts = {}
    train_file = open(data_path + "sorted_filtered_data_remove_time.txt")
    line_number = 0
    for line in train_file:
        line_number += 1
        if line_number % 10000 == 0:
            print "Line number: ", line_number
        list_items = []
        line = line.strip()
        elements = line.split("|")
        uid = int(elements[0])
        # list items
        itemid_string = elements[1].strip()[1:-1]
        itemid_string = itemid_string.split(",")
        for i in itemid_string:
            list_items.append(int(i.strip()))
        dict_items[uid] = list_items

        # list interacts
        list_interact = []
        interact_string = elements[2].strip()[1:-1]
        interact_string = interact_string.split(",")
        for i in interact_string:
            list_interact.append(int(i.strip()))
        dict_iteracts[uid] = list_interact

    new_dict_items = {}
    new_dict_interacts = {}
    for uid in dict_items:  # moi user trong danh sach dict item
        if uid in list_test:  # neu uid do co trong list test
            for k in range(len(dict_items[uid])):  # doi voi moi user
                if dict_items[uid][k] == list_test[uid]:
                    continue
                else:
                    # dict item
                    if uid in new_dict_items:
                        new_dict_items[uid].append(dict_items[uid][k])
                    else:
                        new_dict_items[uid] = [dict_items[uid][k]]
                    # dict interacts
                    if uid in new_dict_interacts:
                        new_dict_interacts[uid].append(dict_iteracts[uid][k])
                    else:
                        new_dict_interacts[uid] = [dict_iteracts[uid][k]]
        else:
            new_dict_items[uid] = dict_items[uid]
            new_dict_interacts[uid] = dict_iteracts[uid]
    # writing into new file train & test
    print("Init writing file")
    file_train = data_path + 'ratings_train_without_implicit.txt'
    write_file = open(file_train, "w")
    writer = csv.writer(write_file, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
    for uid in new_dict_items:
        writer.writerow([uid, new_dict_items[uid], new_dict_interacts[uid]])
    print "Done"


create_train_file_without_implicit_interact()
