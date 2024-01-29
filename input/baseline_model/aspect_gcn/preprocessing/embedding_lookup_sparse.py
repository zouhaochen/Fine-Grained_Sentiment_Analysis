# demo for tesorflow_lookup_sparse.
import numpy as np
import tensorflow as tf

# embedding matrix
example = np.arange(24).reshape(6, 4).astype(np.float32)
embedding = tf.Variable(example)
# embedding lookup SparseTensor
# tao tensor vector voi cac phan tu tai (0,0) = 0
# (0,1) =1 ; (1,1) = 2 ; (1,2) = 3, (2, 0) = 0
# cac phan tu con lai la None
idx = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 1], [1, 2], [2, 0]],
                      values=[0, 1, 2, 3, 0],
                      dense_shape=[3, 3])
weight = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 1], [1, 2], [2, 0]],
                         values=[0.5, 0.5, 2, 3, 1],
                         dense_shape=[3, 3])
# cong cac phan tu trong tensor

embed_sum_no_weight = tf.nn.embedding_lookup_sparse(params=embedding, sp_ids=idx, sp_weights=None, combiner='sum')
embed_sum_weight = tf.nn.embedding_lookup_sparse(params=embedding, sp_ids=idx, sp_weights=weight, combiner='sum')
embed_sum_weight_mean = tf.nn.embedding_lookup_sparse(params=embedding, sp_ids=idx, sp_weights=weight, combiner='mean')
embed_mean = tf.nn.embedding_lookup_sparse(embedding, idx, None, combiner='mean')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Embedding: ")
    print(sess.run(embedding))  # 6 x 4
    print('Sparse vector idx: ')
    print(sess.run(tf.sparse_tensor_to_dense(idx)))  # sparse tensor 3 x3

    print('Weight: ')
    print(sess.run(tf.sparse_tensor_to_dense(weight)))  # sparse tensor 3 x3

    print("Embedded value sum no weight: ")
    print(sess.run(embed_sum_no_weight))

    print("Embedded value sum with weight: ")
    print(sess.run(embed_sum_weight))

    print("Embedded value sum with weight mean: ")
    print(sess.run(embed_sum_weight_mean))
