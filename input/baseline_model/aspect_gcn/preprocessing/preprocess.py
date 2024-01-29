from src import settings
import csv

root_path = settings.DATA_ROOT_PATH + 'site_data/z_recobell/'
data_file = root_path + "_combine.filter.sorted"


def create_sequences():  # split data into train, test.
    retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/z_recobell/'
    data_file = open(retail_rocket_root_path + "_combine.filter.sorted", "r")

    final_sequence = retail_rocket_root_path + "ratings_new_ids.txt"

    interact_item_sequences = {}  # cac chuoi item sequences tuong ung cho moi user
    interact_sequences = {}  # dict user_id va sequence interact tuong ung.
    test_explicit = {}
    new_item_sequences = {}
    new_iteract_sequences = {}
    # thong tin tuong ung voi cac sp:
    # 0: co tuong tac implicit
    # 1: co tuong tac explicit

    data_reader = csv.reader(data_file, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    k = 0
    for line in data_reader:
        if k % 10000:
            print(k)
        uid = int(line[0])
        iid = int(line[1])
        interact = int(line[3])

        # add item vao chuoi sequences item interact.
        if uid in interact_item_sequences:
            interact_item_sequences[uid].append(iid)
        else:
            interact_item_sequences[uid] = [iid]

        # tao chuoi interact tuong ung cua user
        if uid in interact_sequences:
            interact_sequences[uid].append(0)
        else:
            interact_sequences[uid] = [0]

        if interact == 2:  # trong truong hop neu la explicit thi se duoc tach ra 2 tuong tac implicit va explicit
            interact_item_sequences[uid].append(iid)
            interact_sequences[uid].append(1)

    print("Init writing for training")
    writefile = open(final_sequence, "w")
    csv_writer = csv.writer(writefile, delimiter='|', quotechar='"', quoting=csv.QUOTE_NONE)
    for uid in new_item_sequences:
        csv_writer.writerow([uid, interact_item_sequences[uid], interact_sequences[uid]])


create_sequences()
