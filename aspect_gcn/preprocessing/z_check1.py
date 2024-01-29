# writing into train file.
import csv
from src import settings

retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/retail_rocket/'
file1 = open(retail_rocket_root_path + "ratings_new_ids.txt", "r")

train_file = retail_rocket_root_path + "ratings_train_v6.txt"
test_file = retail_rocket_root_path + "ratings_test_v6.txt"

number_explicit = 0
train_items = {}
train_interacts = {}
test_item = {}
number_line = 0

count = 0
for line in file1.readlines():
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
        itemids.append(int(item_id.strip()))  # list

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
        itemtest = itemids[flag]
        test_item[int(elements[0].strip())] = itemtest

        # xoa bo phan tu do trong tap du lieu train (chu y co the nhieu hon 1 lan mua)
        list_item_id = []
        list_interact_id = []

        for _ in range(len(itemids)):
            # check bo di nhung item co hanh dong mua ma da duoc dua vao test.
            # giu va bo di tuong tac click
            if (itemids[_] == itemtest) and (interactids[_] == 1):  # xoa tat cac item test co interact 1.
                continue
            # elif (itemids[_] == itemtest) and (_ == flag - 1):  # bo tuong tac click ngay truoc do.
            #     continue
            elif itemids[_] == itemtest:  # bo di tat ca cac tuong tac click.
                count += 1
                continue
            else:
                list_item_id.append(itemids[_])
                list_interact_id.append(interactids[_])
        train_items[int(elements[0].strip())] = list_item_id
        train_interacts[int(elements[0].strip())] = list_interact_id
    else:
        train_items[int(elements[0].strip())] = itemids
        train_interacts[int(elements[0].strip())] = interactids
# print "Dict:", train
print "remove: ", count
print("Last: ", len(train_items.keys()))

print("Init writing train file")
writefile = open(train_file, "w")
csv_writer = csv.writer(writefile, delimiter='|', quotechar='', quoting=csv.QUOTE_NONE)
for uid in train_items:
    csv_writer.writerow([uid, train_items[uid], train_interacts[uid]])

print("Init writing test file")
writefile = open(test_file, "w")
csv_writer = csv.writer(writefile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
for uid in test_item:
    csv_writer.writerow([uid, test_item[uid]])
