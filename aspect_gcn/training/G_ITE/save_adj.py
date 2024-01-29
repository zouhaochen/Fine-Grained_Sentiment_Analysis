import sys
sys.path.append('code/jounal_ite')
from src.data_preparation import data_utils
import scipy.sparse as sp

root_path = "datasets/recobell/"
num_user, num_item = data_utils.load_representation_data(root_path + "u2index.txt",root_path + "i2index.txt")

for threshold in [200, 500, 700, 1000]:
    user_adj, item_adj = data_utils.load_adj_mat(root_path + "without_implicit_in_train/" + "_train.rating", num_user, num_item, threshold)
    sp.save_npz(root_path + "without_implicit_in_train/" + "user_adj_"+str(threshold)+".npz", user_adj)
    sp.save_npz(root_path + "without_implicit_in_train/" + "item_adj_"+str(threshold)+".npz", item_adj)
    print("1 done")
