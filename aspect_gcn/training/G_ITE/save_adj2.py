import sys
sys.path.append('code/jounal_ite')
from src.data_preparation import data_utils
import scipy.sparse as sp
import numpy as np
import h5py

root_path = "datasets/t_mall/"
num_user, num_item = data_utils.load_representation_data(root_path + "u2index.txt",root_path + "i2index.txt")
int_interact_mat = data_utils.load_int_interact_matrix(root_path + "without_implicit_in_train/" + "_train.rating", num_user, num_item)

num_batch = 10
u_batch = int(num_user/num_batch)
i_batch = int(num_item/num_batch)

all_item_adj=None
all_user_adj=None

for i in range(num_batch+1):
    u_start=i*u_batch
    u_end=(i+1)*u_batch
    if u_end>num_user:
        u_end=num_user

    i_start = i * i_batch
    i_end = (i + 1) * i_batch
    if i_end > num_item:
        i_end = num_item


    user_adj = int_interact_mat[u_start:u_end,:].dot(int_interact_mat.transpose()).tocoo()
    user_adj, user_deg = data_utils.construct_adj(row=user_adj.row, col=user_adj.col,
         rate=user_adj.data, num=user_adj.shape[0], max_deg=10, offset=u_start)

    item_adj = int_interact_mat.transpose()[i_start:i_end,:].dot(int_interact_mat).tocoo()
    item_adj, item_deg = data_utils.construct_adj(row=item_adj.row, col=item_adj.col,
        rate=item_adj.data, num=item_adj.shape[0], max_deg=10, offset=i_start)

    if all_user_adj is None:
        all_user_adj = user_adj
    else: all_user_adj = np.vstack([all_user_adj, user_adj])

    if all_item_adj is None:
        all_item_adj = item_adj
    else: all_item_adj = np.vstack([all_item_adj, item_adj])

    print("done ", i)
hf = h5py.File('data.h5', 'w')
hf.create_dataset('user_adj_full', data=all_user_adj)
hf.create_dataset('item_adj_full', data=all_item_adj)
hf.close()
print(all_user_adj.shape, all_item_adj.shape)