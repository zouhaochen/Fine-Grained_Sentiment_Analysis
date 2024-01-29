import tensorflow as tf
from GCN.training.G_ITE import aggregators
from numpy.random import seed
seed(1)
tf.random.set_random_seed(seed=2)

class ITE:
    @staticmethod
    def get_place_holder():
        user_indices = tf.placeholder(dtype=tf.int64, name="user_indices")
        item_indices = tf.placeholder(dtype=tf.int64, name="item_indices")

        return user_indices, item_indices

    @staticmethod
    def get_embedding_weight(user_adj_mat, item_adj_mat):
        """
        num_factor: number of factor in the last hidden layer of GMF and MLP part
        Refer to model_ite_onehot_log_loss architecture for better understand the values of num_factor_gmf and num_factor_mlp
        """
        user_adj = tf.constant(value=user_adj_mat, name="user_adj", dtype=tf.int64)
        item_adj = tf.constant(value=item_adj_mat, name="item_adj", dtype=tf.int64)

        return {
                "user_adj": user_adj,
                "item_adj": item_adj
                }

    @staticmethod
    def create_model(hyper_params):
        # custom hyper_params
        num_user = hyper_params["num_user"]
        num_item = hyper_params["num_item"]
        learning_rate = hyper_params["lr"]
        num_factor = hyper_params["num_factor"]
        eta = hyper_params["eta"]
        q_lambda = hyper_params["lambda"]
        user_adj_mat = hyper_params['user_adj']
        item_adj_mat = hyper_params['item_adj']
        num_sample = hyper_params['num_sample']
        max_deg = hyper_params['max_deg']

        # with tf.device("/gpu:0"):
        user_indices, item_indices = ITE.get_place_holder()
        embedding_weight = ITE.get_embedding_weight(user_adj_mat, item_adj_mat)
        num_factor_gmf = num_factor
        num_factor_mlp = 2 * num_factor

        # ------------------------------- Sample ----------------------------------
        user_samples = [user_indices]

        user_node = tf.nn.embedding_lookup(embedding_weight["user_adj"], user_samples[-1])
        user_node = tf.transpose(tf.random_shuffle(tf.transpose(user_node)))
        user_node = tf.slice(user_node, [0, 0], [-1, num_sample])
        user_samples.append(user_node)

        user_node = tf.nn.embedding_lookup(embedding_weight["user_adj"], user_samples[-1])
        user_node = tf.reshape(user_node, [-1, max_deg])
        user_node = tf.transpose(tf.random_shuffle(tf.transpose(user_node)))
        user_node = tf.slice(user_node, [0, 0], [-1, num_sample])
        user_node = tf.reshape(user_node, [-1, num_sample, num_sample])
        user_samples.append(user_node)

        item_samples = [item_indices]

        item_node = tf.nn.embedding_lookup(embedding_weight["item_adj"], item_samples[-1])
        item_node = tf.transpose(tf.random_shuffle(tf.transpose(item_node)))
        item_node = tf.slice(item_node, [0, 0], [-1, num_sample])
        item_samples.append(item_node)

        item_node = tf.nn.embedding_lookup(embedding_weight["item_adj"], item_samples[-1])
        item_node = tf.reshape(item_node, [-1, max_deg])
        item_node = tf.transpose(tf.random_shuffle(tf.transpose(item_node)))
        item_node = tf.slice(item_node, [0, 0], [-1, num_sample])
        item_node = tf.reshape(item_node, [-1, num_sample, num_sample])
        item_samples.append(item_node)

        # ------------------------------- Aggregate ----------------------------------
        gmf_user_agg = aggregators.GCNAggregator_1layer(input_dim=num_user, output_dim=num_factor_gmf, name="gmf_weight_user")
        gmf_user_vec = gmf_user_agg(samples=user_samples)

        gmf_item_agg = aggregators.GCNAggregator_1layer(input_dim=num_item, output_dim=num_factor_gmf, name="gmf_weight_item")
        gmf_item_vec = gmf_item_agg(samples=item_samples)

        mlp_user_agg = aggregators.GCNAggregator_1layer(input_dim=num_user, output_dim=num_factor_mlp, name="mlp_weight_user")
        mlp_user_vec = mlp_user_agg(samples=user_samples)

        mlp_item_agg = aggregators.GCNAggregator_1layer(input_dim=num_item, output_dim=num_factor_mlp, name="mlp_weight_item")
        mlp_item_vec = mlp_item_agg(samples=item_samples)

        gcn_weights = []
        gcn_weights.extend(gmf_user_agg.get_weight())
        gcn_weights.extend(gmf_item_agg.get_weight())
        gcn_weights.extend(mlp_user_agg.get_weight())
        gcn_weights.extend(mlp_item_agg.get_weight())


        # -------------------------------- GMF part -------------------------------
        # gmf_pu_onehot = tf.nn.embedding_lookup(embedding_weight["gmf_user_onehot"], user_indices,
        #                                        name="gmf_pu_onehot")
        # gmf_qi_onehot = tf.nn.embedding_lookup(embedding_weight["gmf_item_onehot"], item_indices,
        #                                        name="gmf_qi_onehot")
        gmf_pu = tf.identity(gmf_user_vec, name="gmf_pu")
        gmf_qi = tf.identity(gmf_item_vec, name="gmf_qi")

        gmf_phi = tf.multiply(gmf_pu, gmf_qi, name="gmf_phi")
        gmf_h = tf.Variable(tf.random_uniform([num_factor, 1], minval=-1, maxval=1), name="gmf_h")


        # --------------------------------- MLP part --------------------------------
        # mlp_pu_onehot = tf.nn.embedding_lookup(embedding_weight["mlp_user_onehot"], user_indices,
        #                                        name="mlp_pu_onehot")
        # mlp_qi_onehot = tf.nn.embedding_lookup(embedding_weight["mlp_item_onehot"], item_indices,
        #                                        name="mlp_qi_onehot")

        mlp_pu = tf.identity(mlp_user_vec, name="mlp_pu")
        mlp_qi = tf.identity(mlp_item_vec, name="mlp_qi")

        mlp_weights = {
            "w1": tf.Variable(tf.random_normal([4 * num_factor, 2 * num_factor]) * tf.sqrt(1 / num_factor),
                              name="mlp_weight1"),
            "w2": tf.Variable(tf.random_normal([2 * num_factor, num_factor]) * tf.sqrt(2 / num_factor),
                              name="mlp_weight2"),
            "h": tf.Variable(tf.random_uniform([num_factor, 1], minval=-1, maxval=1), name="mlp_h")
        }
        mlp_biases = {
            "b1": tf.Variable(tf.random_normal([2 * num_factor]), name="mlp_bias1"),
            "b2": tf.Variable(tf.random_normal([num_factor]), name="mlp_bias2")
        }

        mlp_phi_1 = tf.concat([mlp_pu, mlp_qi], axis=-1, name="mlp_phi1")
        mlp_phi_2 = tf.nn.leaky_relu(tf.add(tf.matmul(mlp_phi_1, mlp_weights["w1"]), mlp_biases["b1"]),
                                     name="mlp_phi2")
        mlp_phi_3 = tf.nn.leaky_relu(tf.add(tf.matmul(mlp_phi_2, mlp_weights["w2"]), mlp_biases["b2"]),
                                     name="mlp_phi3")

        # --------------------------------- implicit part ------------------------------------
        # 1 x 2*num_factor
        im_phi = tf.concat([gmf_phi, mlp_phi_3], axis=1, name="im_phi")
        im_bias = tf.Variable(0.0, name="im_bias")
        # 2*num_factor x 1
        h_implicit = tf.concat([gmf_h, mlp_weights["h"]], axis=0, name="h_implicit")
        # tf.squeeze() 1 x 1
        train_im_prediction = tf.squeeze(tf.add(tf.matmul(im_phi, h_implicit), im_bias), name="train_im_prediction")

        # --------------------------------- explicit part ------------------------------------
        im_phi_2 = tf.concat([gmf_user_vec, mlp_user_vec,
                              im_phi,
                              gmf_item_vec, mlp_item_vec],
                             axis=1, name="im_phi_2")

        im2ex_weights = {
            "w1": tf.Variable(tf.random_normal([8 * num_factor, num_factor]) * tf.sqrt(2 / num_factor),
                              name="im2ex_weight1"),
            "h": tf.Variable(tf.random_uniform([num_factor, 1], minval=-1, maxval=1), name="explicit_h")
        }
        im2ex_biases = {
            "b1": tf.Variable(tf.random_normal([num_factor]), name="im2ex_bias1"),
        }
        # 1 x num_factor
        ex_phi = tf.nn.leaky_relu(tf.add(tf.matmul(im_phi_2, im2ex_weights["w1"]), im2ex_biases["b1"]), name="ex_phi")
        ex_bias = tf.Variable(0.0, name="ex_bias")
        train_ex_prediction = tf.squeeze(tf.add(tf.matmul(ex_phi, im2ex_weights["h"]), ex_bias),
                                         name="train_ex_prediction")
        # ex_prediction = tf.squeeze(tf.multiply(train_im_prediction, train_ex_prediction), name="ex_prediction")
        ex_prediction = tf.squeeze(
            tf.multiply(tf.nn.sigmoid(train_im_prediction), tf.nn.sigmoid(train_ex_prediction)),
            name="ex_prediction")
        # ex_prediction = tf.squeeze(tf.nn.sigmoid(train_ex_prediction), name="ex_prediction")

        """
        # ---------------------------------- square loss ---------------------------------------------
        labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        y1_indicators = tf.placeholder(tf.float32, shape=[None], name="y1_indicators")
        y2_indicators = tf.placeholder(tf.float32, shape=[None], name="y2_indicators")

        loss_implicit_list = tf.square(tf.subtract(labels, train_im_prediction), name="y1_loss_list")
        loss_implicit = tf.reduce_mean(tf.multiply(y1_indicators, loss_implicit_list), name="y1_loss")
        loss_explicit_list = tf.square(tf.subtract(labels, train_ex_prediction), name="y2_loss_list")
        loss_explicit = tf.reduce_mean(tf.multiply(y2_indicators, loss_explicit_list), name="y2_loss")

        regularizer = tf.add(tf.add(tf.reduce_mean(tf.square(gmf_pu)), tf.reduce_mean(tf.square(gmf_qi))),
                             tf.add(tf.reduce_mean(tf.square(mlp_pu)), tf.reduce_mean(tf.square(mlp_qi))),
                             name="regularizer")

        loss = tf.add(tf.add(tf.multiply(eta_1, loss_implicit), loss_explicit), tf.multiply(q_lambda, regularizer),
                      name="loss")
        """
        # ---------------------------------- log loss ---------------------------------------------
        labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        y1_indicators = tf.placeholder(tf.float32, shape=[None], name='y1_indicators')
        y2_indicators = tf.placeholder(tf.float32, shape=[None], name='y2_indicators')

        loss_implicit_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                     logits=train_im_prediction,
                                                                     name="y1_loss_list")
        loss_implicit = tf.reduce_mean(tf.multiply(y1_indicators, loss_implicit_list), name='y1_loss')
        loss_explicit_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                     logits=train_ex_prediction,
                                                                     name='y2_loss_list')
        loss_explicit = tf.reduce_mean(tf.multiply(y2_indicators, loss_explicit_list), name='y2_loss')

        gcn_weights_mean_square = [tf.reduce_mean(tf.square(weight)) for weight in gcn_weights]
        regularizer = tf.add_n(gcn_weights_mean_square, name="regularizer")
        # regularizer = tf.add(tf.add(tf.reduce_mean(tf.square(gmf_pu)), tf.reduce_mean(tf.square(gmf_qi))),
        #                      tf.add(tf.reduce_mean(tf.square(mlp_pu)), tf.reduce_mean(tf.square(mlp_qi))),
        #                      name="regularizer")

        test_loss = tf.add(tf.multiply(eta, loss_explicit), tf.multiply(1.0, loss_implicit), name="test_loss")
        train_loss = tf.add(test_loss, tf.multiply(q_lambda, regularizer), name='train_loss')

        # optimize
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss, name="optimizer")

        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_implicit, name="optimize")
        # optimizer = tf.train.MomentumOptimizer(0.0001, 0.8).minimize(loss, name="optimize")

        return {
            "user_indices_ph": user_indices,
            "item_indices_ph": item_indices,
            "labels_ph": labels,
            "y1_indicators_ph": y1_indicators,
            "y2_indicators_ph": y2_indicators,
            "optimizer": optimizer,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "prediction": ex_prediction,
            "mlp_phi_3": mlp_phi_3,
            "train_im_prediction": train_im_prediction
            # "global_epoch": global_epoch
        }
