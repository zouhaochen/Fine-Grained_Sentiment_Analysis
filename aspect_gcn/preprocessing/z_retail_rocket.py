import csv
from src import settings

retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/ml-1m/'

# from _implicit.filter.clean.txt, _explicit.filter.clean.txt merge into one name z_temp_ratings.txt.
implicit_file = retail_rocket_root_path + "_implicit.filter.clean.txt"
explicit_file = retail_rocket_root_path + "_explicit.filter.clean.txt"


def get_implicit_interact_dict():
    implicit_interacted_dict = {}
    file_pointer = open(implicit_file, "r")  # reading from _implicit.clean.txt
    csv_reader = csv.reader(file_pointer, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in csv_reader:
        raw_uid = line[0]  # string
        raw_item_id = line[1]
        time = line[2]
        if raw_uid in implicit_interacted_dict:
            implicit_interacted_dict[raw_uid].append([raw_item_id, time])
        else:
            implicit_interacted_dict[raw_uid] = [[raw_item_id, time]]
    return implicit_interacted_dict


def get_explicit_interact_dict():
    explicit_interacted_dict = {}
    file_pointer = open(explicit_file, "r")  # reading from _implicit.clean.txt
    csv_reader = csv.reader(file_pointer, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in csv_reader:
        raw_uid = line[0]  # string
        raw_item_id = line[1]
        time = line[2]
        if raw_uid in explicit_interacted_dict:
            explicit_interacted_dict[raw_uid].append([raw_item_id, time])
        else:
            explicit_interacted_dict[raw_uid] = [[raw_item_id, time]]
    return explicit_interacted_dict


# get list user in explicit.
def get_list_explicit_user():
    list_explicit_user = []
    file_pointer = open(explicit_file, "r")
    csv_reader = csv.reader(file_pointer, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in csv_reader:
        raw_uid = line[0]  # string
        if raw_uid not in list_explicit_user:
            list_explicit_user.append(raw_uid)
    return list_explicit_user


# get list user in implicit.
def get_list_implicit_user():
    list_implicit_user = []
    file_pointer = open(implicit_file, "r")
    csv_reader = csv.reader(file_pointer, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for line in csv_reader:
        raw_uid = line[0]  # string
        if raw_uid not in list_implicit_user:
            list_implicit_user.append(raw_uid)
    return list_implicit_user


# merge two types of data
print("Get list implicit and explicit user")
list_explicit_user = get_list_explicit_user()
list_implicit_user = get_list_implicit_user()

print("Get dict implicit and explicit user")
implicit_interacted_dict = get_implicit_interact_dict()
explicit_interacted_dict = get_explicit_interact_dict()
# output file to write

print("Init writing")
output = retail_rocket_root_path + 'z_temp_ratings.csv'
writefile = open(output, "w")
csv_writer = csv.writer(writefile, delimiter='|', quotechar='"', quoting=csv.QUOTE_NONE)

print("processing on implicit dict")
for uid in implicit_interacted_dict:
    if uid in list_explicit_user:
        csv_writer.writerow([uid, implicit_interacted_dict[uid], explicit_interacted_dict[uid]])
    else:
        csv_writer.writerow([uid, implicit_interacted_dict[uid], -1])
        exit()
# dua ra not truong hop con lai, khong co trong implicit ma co trong explicit.
print("processing on explicit dict")
for uid in explicit_interacted_dict:
    if uid not in list_implicit_user:
        csv_writer.writerow([uid,-1, explicit_interacted_dict[uid]])
print("done")
