import sys
sys.path.append('code/jounal_ite')
from src.data_preparation import data_utils
import scipy.sparse as sp
import numpy as np
import h5py


root_path = "datasets/recobell/"
with h5py.File(root_path + "without_implicit_in_train/data.h5", "r") as f:
    user_adj_full = np.array(f['user_adj_full'])
    item_adj_full = np.array(f['item_adj_full'])

num_user, num_item = data_utils.load_representation_data(root_path + "u2index.txt",root_path + "i2index.txt")
int_interact_mat = data_utils.load_int_interact_matrix(root_path + "without_implicit_in_train/" + "_train.rating", num_user, num_item)

user_adj = int_interact_mat[206400:,:].dot(int_interact_mat.transpose()).tocoo()
user_adj, user_deg = data_utils.construct_adj(row=user_adj.row, col=user_adj.col,
    rate=user_adj.data, num=user_adj.shape[0], max_deg=10)

item_adj = int_interact_mat.transpose()[118440:,:].dot(int_interact_mat).tocoo()
item_adj, item_deg = data_utils.construct_adj(row=item_adj.row, col=item_adj.col,
    rate=item_adj.data, num=item_adj.shape[0], max_deg=10)

user_adj_full = np.vstack([user_adj_full, user_adj])
item_adj_full = np.vstack([item_adj_full, item_adj])
hf = h5py.File('data2.h5', 'w')
hf.create_dataset('user_adj_full', data=user_adj_full)
hf.create_dataset('item_adj_full', data=item_adj_full)
hf.close()
print(user_adj_full.shape, item_adj_full.shape)

