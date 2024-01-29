# writing into train file.
import csv
from src import settings

retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/retail_rocket/'
file1 = open(retail_rocket_root_path + "z_ratings_test_v7.txt", "r")
file2 = retail_rocket_root_path + "ratings_test_v7.txt"

train_items = {}
train_interacts = {}
test_item = {}
number_line = 0
list_item = {}
list1 = {}

for line in file1.readlines():
    number_line += 1
    if number_line % 10000 == 0:
        print("Line: ", number_line, "/", 36751)

    elements = line.split("|")
    # print "Elements: ", elements
    # exit()
    user_id = int(elements[0])
    item = int(elements[1])
    list1[user_id] = item

with open(file2, 'w') as f:  # viet ra file 'i2index.txt'
    csv_writer = csv.writer(f, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
    for k in list1:
        csv_writer.writerow([k, list1[k]])  # item id raw => itemid moi.

print "Done"
