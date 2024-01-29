from src import settings
import csv

retail_rocket_root_path = settings.DATA_ROOT_PATH + 'site_data/retail_rocket/'
file1 = open(retail_rocket_root_path + "z_temp_ratings.txt", "r")

final_ratings = retail_rocket_root_path + "z_ratings.txt"


# from z_temp_ratings.txt => z_ratings.txt
def Sort(sub_li):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    sub_li.sort(key=lambda x: x[1])
    return sub_li


interact_item_sequences = {}  # cac chuoi item sequences tuong ung cho moi
interact_sequences = {}
# thong tin tuong ung voi cac sp:
# 0: chi co tuong tac implicit
# 1: chi co tuong tac explicit
# 2: co ca implicit & explicit

for line in file1.readlines():
    line = line.strip()
    print("Line:", line)
    elements = line.split("|")
    # user id
    uid = elements[0].strip()
    # explicit interact
    explicit_interact = elements[2].strip()

    if explicit_interact == '-1':  # khong co tuong tac explicit
        # xu ly implicit interact.
        implicit_interacts = elements[1].strip()
        implicit_interacts = implicit_interacts[2:-2]  # loai bo 2 phan tu dau va cuoi

        list_item = []
        for implicit_pair in implicit_interacts.split("],"):
            implicit_pair = implicit_pair.strip()
            if '[' in implicit_pair:
                implicit_pair = implicit_pair[1:]
            itemid = implicit_pair.split(",")[0].strip()[1:-1]
            time = implicit_pair.split(",")[1].strip()[1:-1]
            list_item.append(int(itemid))
            print(itemid + "-" + time)
        # tong hop lai vao dict[uid]
        interact_item_sequences[uid] = list_item
        interact_sequences[uid] = [0] * len(list_item)  # mang gom list cac phan tu = 0

    else:  # neu co tuong tac explicit
        # xu ly implicit interact.
        implicit_interacts = elements[1].strip()
        #         print "ban dau:"
        #         print implicit_interacts
        implicit_interacts = implicit_interacts[2:-2]  # loai bo 2 phan tu dau va cuoi

        list_implicit = []
        list_item_implicit = []
        for implicit_pair in implicit_interacts.split("],"):
            implicit_pair = implicit_pair.strip()
            if '[' in implicit_pair:
                implicit_pair = implicit_pair[1:]
            itemid = implicit_pair.split(",")[0].strip()[1:-1]
            time = implicit_pair.split(",")[1].strip()[1:-1]

            list_implicit.append([itemid + 'i', int(time)])
        #         print "Sau do:"
        #         print Sort(list_implicit)

        # with explicit interact.
        explicit_interacts = elements[2].strip()
        explicit_interacts = explicit_interacts[2:-2]  # loai bo 2 phan tu dau va cuoi
        list_explicit = []
        list_item_explicit = []
        for explicit_pair in explicit_interacts.split("],"):
            explicit_pair = explicit_pair.strip()
            if '[' in explicit_pair:
                explicit_pair = explicit_pair[1:]
            itemid = explicit_pair.split(",")[0].strip()[1:-1]
            time = explicit_pair.split(",")[1].strip()[1:-1]

            list_explicit.append([itemid + 'e', int(time)])

        total_list = list_implicit + list_explicit

        # list item
        total_item = []
        for _ in total_list:
            if int(_[0][:-1]) not in total_item:
                total_item.append(int(_[0][:-1]))

        # sorting by time:
        sorted_list = Sort(total_list)

        # tong hop lai vao dict[uid]
        list_temp1 = []
        list_temp2 = []
        if len(total_item) == len(sorted_list):  # neu so item xh == so ban ghi
            for element in sorted_list:
                list_temp1.append(int(element[0][:-1]))
                if element[0][-1] == 'i':
                    list_temp2.append(0)
                else:
                    list_temp2.append(1)
            interact_item_sequences[uid] = list_temp1
            interact_sequences[uid] = list_temp2

        # neu ngan hon, hay tuc la co 1 item duoc tuong tac tai cac thoi diem khac nhau.
        # list_item of sorted list (include i and e)
        # print("Sorted list:")
        # for _ in sorted_list:
        #     print(_)

        list_item_with_status = []
        list_item_without_status = []
        for _ in sorted_list:
            list_item_with_status.append(_[0])
            list_item_without_status.append(int(_[0][:-1]))

        if len(total_item) < len(sorted_list):
            temp_add_list_with_status = []
            new_sort = []
            for element in sorted_list:
                if element[0][-1] == 'i':  # neu la implicit
                    new_sort.append([element[0], 0])

                elif element[0][-1] == 'e':  # neu la explicit
                    if temp_add_list_with_status.count(element[0].replace(element[0][-1], 'i')) > 0:
                        #                         print "TEMP STT:",temp_add_list_with_status
                        # ma da co tuong tac implicit truoc do
                        # cap nhat thang implicit gan nhat tu 0 thanh 2.
                        for _ in reversed(new_sort):
                            if _[0] == element[0].replace(element[0][-1], 'i'):
                                _[1] = 2
                                break
                    else:
                        new_sort.append([element[0], 1])
                temp_add_list_with_status.append(element[0])
            # print("New sorted:")
            # for _ in new_sort:
            #     print(_)

            temp_1 = []
            temp_2 = []
            for _ in new_sort:  # ghi vao list interact
                temp_1.append(int(_[0][:-1]))
                temp_2.append(_[1])
            interact_item_sequences[uid] = temp_1
            interact_sequences[uid] = temp_2

    # writing into file text

    # print(interact_item_sequences)
    # print(interact_sequences)

    print("Init writing")
    writefile = open(final_ratings, "w")
    csv_writer = csv.writer(writefile, delimiter='|', quotechar='"', quoting=csv.QUOTE_NONE)
    for uid in interact_item_sequences:
        csv_writer.writerow([uid, interact_item_sequences[uid], interact_sequences[uid]])
