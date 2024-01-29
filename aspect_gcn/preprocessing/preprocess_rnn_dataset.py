# writing into train file.
import csv
from src import settings

retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/recobell/'
file1 = open(retail_rocket_root_path + "ratings_new_ids.txt", "r")

# writing into train and test for rnn model.
train_file = retail_rocket_root_path + "rnn/ratings_train_rnn.txt"
test_file = retail_rocket_root_path + "rnn/ratings_test_rnn.txt"

number_explicit = 0
train_items = {}
train_interacts = {}
test_item = {}
number_line = 0

for line in file1.readlines():  # voi moi dong trong ratings_new_ids.txt
    number_line += 1
    if number_line % 10000 == 0:
        print("Line: ", number_line, "/", 36751)
    flag = -1  # danh dau vi tri last explicit.
    line = line.strip()
    elements = line.split("|")

    item_ids = elements[1].strip()[1:-1]
    itemids = []
    interactids = []

    for item_id in item_ids.split(","):
        itemids.append(int(item_id.strip()))  # convert into list itemids

    interacts = elements[2].strip()[1:-1]  # xu ly mang cac interact.
    for interact in interacts.split(","):
        val = interact.strip()
        interactids.append(int(val))  # list

    if len(itemids) != len(interactids):  # neu 2 chuoi co do dai khac nhau.
        continue

    # tim cap u-i dua vao test
    for i in range(len(interactids)):
        if interactids[i] == 1:  # duyet va gan lay phan tu cuoi cung
            flag = i

    if flag != -1:  # co gia tri duoc dua vao test
        # doi voi vi tri moi tim duoc
        # cap test item
        test_item[int(elements[0].strip())] = [itemids[flag - 1], itemids[flag]]
        # xoa bo item do trong tap du lieu train (chu y co the nhieu hon 1 lan mua)
        list_item_id = []
        list_interact_id = []
        for _ in range(len(itemids)):
            # bo di item tai vi tri da duoc chon vao test va item ngay truoc no
            if _ == flag - 1:  # ngay truoc item chon vao test
                continue
            elif _ == flag:  # item test
                continue
            else:
                list_item_id.append(itemids[_])

        train_items[int(elements[0].strip())] = list_item_id
    else:
        train_items[int(elements[0].strip())] = itemids

# print "Dict:", train
print("Last: ", len(train_items.keys()))

print("Init writing train file for rnn model")
writefile = open(train_file, "w")
csv_writer = csv.writer(writefile, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
for uid in train_items:
    csv_writer.writerow([uid, train_items[uid]])

print("Init writing test file for rnn model")
writefile = open(test_file, "w")
csv_writer = csv.writer(writefile, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
for uid in test_item:
    csv_writer.writerow([uid, test_item[uid]])
