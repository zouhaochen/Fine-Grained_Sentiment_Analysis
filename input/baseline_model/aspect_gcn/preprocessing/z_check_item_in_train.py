# writing into train file.
from src import settings

retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/retail_rocket/z_data_processing/'
# file1 = open(retail_rocket_root_path + "ratings_train_without_implicit.txt", "r")
file1 = open(retail_rocket_root_path + "ratings_train_with_implicit.txt", "r")
file2 = open(retail_rocket_root_path + "ratings_test.txt", "r")

number_line = 0
list_train = {}
count = 0
for line in file1.readlines():
    number_line += 1
    if number_line % 10000 == 0:
        print("Line: ", number_line, "/", 36751)
    flag = -1  # danh dau vi tri last explicit.
    line = line.strip()
    elements = line.split("|")
    # print "Elements: ", elements
    user_id = int(elements[0].strip())
    item_ids = elements[1].strip()[1:-1]
    itemids = []

    for item_id in item_ids.split(","):
        itemids.append(int(item_id.strip()))  # list
    list_train[user_id] = itemids
print "Done reading train"

list_test = {}
for line in file2.readlines():
    line = line.strip()
    elements = line.split(",")
    user_id = int(elements[0])
    item_id = int(elements[1])
    list_test[user_id] = item_id
print "Done reading test"

print "Check test item in train"
count = 0
for uid in list_test:
    if list_test[uid] in list_train[uid]:
        print "uid: ", uid
        print "list test: ", list_test[uid]
        print "list train: ", list_train[uid]
        count += 1
print "Count: ", count
