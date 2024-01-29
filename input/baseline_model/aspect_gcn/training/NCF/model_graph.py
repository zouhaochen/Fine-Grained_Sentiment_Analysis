import tensorflow as tf


class NeuMF:
    @staticmethod
    def get_place_holder():
        user_indices = tf.placeholder(dtype=tf.int64, name="user_indices")
        item_indices = tf.placeholder(dtype=tf.int64, name="item_indices")

        return user_indices, item_indices

    @staticmethod
    def get_embedding_weight(num_user, num_item, num_factor):
        """
        num_factor: number of factor in the last hidden layer of GMF and MLP part
        Refer to model_ite_onehot_log_loss architecture for better understand the values of num_factor_gmf and num_factor_mlp
        """

        num_factor_gmf = num_factor
        num_factor_mlp = 2 * num_factor

        gmf_embedding_weight_user_onehot = tf.Variable(
            tf.random_normal([num_user, num_factor_gmf]) * tf.sqrt(2 / num_factor_gmf),
            name="gmf_embedding_weight_user_onehot")
        gmf_embedding_weight_item_onehot = tf.Variable(
            tf.random_normal([num_item, num_factor_gmf]) * tf.sqrt(2 / num_factor_gmf),
            name="gmf_embedding_weight_item_onehot")
        mlp_embedding_weight_user_onehot = tf.Variable(
            tf.random_normal([num_user, num_factor_mlp]) * tf.sqrt(2 / num_factor_mlp),
            name="gmf_embedding_weight_user_onehot")
        mlp_embedding_weight_item_onehot = tf.Variable(
            tf.random_normal([num_item, num_factor_mlp]) * tf.sqrt(2 / num_factor_mlp),
            name="gmf_embedding_weight_item_onehot")

        return {"gmf_user_onehot": gmf_embedding_weight_user_onehot,
                "gmf_item_onehot": gmf_embedding_weight_item_onehot,
                "mlp_user_onehot": mlp_embedding_weight_user_onehot,
                "mlp_item_onehot": mlp_embedding_weight_item_onehot}

    @staticmethod
    def create_model(hyper_params):
        # custom hyper_params
        num_user = hyper_params["num_user"]
        num_item = hyper_params["num_item"]
        learning_rate = hyper_params["lr"]
        num_factor = hyper_params["num_factor"]
        q_lambda = hyper_params["lambda"]
        # global_epoch = tf.Variable(0, dtype=tf.int64, name="global_epoch")

        with tf.device("/gpu:0"):
            user_indices, item_indices = NeuMF.get_place_holder()
            embedding_weight = NeuMF.get_embedding_weight(num_user, num_item, num_factor)

            # -------------------------------- GMF part -------------------------------
            gmf_pu_onehot = tf.nn.embedding_lookup(embedding_weight["gmf_user_onehot"], user_indices,
                                                   name="gmf_pu_onehot")
            gmf_qi_onehot = tf.nn.embedding_lookup(embedding_weight["gmf_item_onehot"], item_indices,
                                                   name="gmf_qi_onehot")
            gmf_pu = tf.identity(gmf_pu_onehot, name="gmf_pu")
            gmf_qi = tf.identity(gmf_qi_onehot, name="gmf_qi")

            gmf_phi = tf.multiply(gmf_pu, gmf_qi, name="gmf_phi")
            gmf_h = tf.Variable(tf.random_uniform([num_factor, 1], minval=-1, maxval=1), name="gmf_h")

            # --------------------------------- MLP part --------------------------------
            mlp_pu_onehot = tf.nn.embedding_lookup(embedding_weight["mlp_user_onehot"], user_indices,
                                                   name="mlp_pu_onehot")
            mlp_qi_onehot = tf.nn.embedding_lookup(embedding_weight["mlp_item_onehot"], item_indices,
                                                   name="mlp_qi_onehot")

            mlp_pu = tf.identity(mlp_pu_onehot, name="mlp_pu")
            mlp_qi = tf.identity(mlp_qi_onehot, name="mlp_qi")

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
            im_prediction = tf.squeeze(tf.nn.sigmoid(train_im_prediction), name="ex_prediction")

            # --------------------------------- explicit part ------------------------------------
            # ex_weights = {
            #     "w1": tf.Variable(tf.random_normal([2 * num_factor, num_factor]) * tf.sqrt(2 / num_factor),
            #                       name="ex_weight1"),
            #     "h": tf.Variable(tf.random_uniform([num_factor, 1], minval=-1, maxval=1), name="h_explicit")
            # }
            # ex_biases = {
            #     "b1": tf.Variable(tf.random_normal([num_factor]), name="ex_bias1"),
            # }
            # # 1 x num_factor
            # ex_phi = tf.nn.leaky_relu(tf.add(tf.matmul(im_phi, ex_weights["w1"]), ex_biases["b1"]), name="ex_phi")
            # train_ex_prediction = tf.squeeze(tf.matmul(ex_phi, ex_weights["h"]), name="train_prediction_explicit")
            # train_ex_prediction = train_im_prediction
            # ex_prediction = train_im_prediction
            # ex_prediction = tf.squeeze(tf.multiply(train_im_prediction, train_ex_prediction), name="prediction_explicit")
            # ex_prediction = tf.squeeze(tf.multiply(tf.nn.sigmoid(train_im_prediction), tf.nn.sigmoid(train_ex_prediction)),
            #                            name="prediction_explicit")
            # ex_prediction = tf.squeeze(tf.nn.sigmoid(train_ex_prediction), name="prediction_explicit")

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

            loss_implicit_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                         logits=train_im_prediction,
                                                                         name="y1_loss_list")
            loss_implicit = tf.reduce_mean(loss_implicit_list, name="y1_loss")

            regularizer = tf.add(tf.add(tf.reduce_mean(tf.square(gmf_pu)), tf.reduce_mean(tf.square(gmf_qi))),
                                 tf.add(tf.reduce_mean(tf.square(mlp_pu)), tf.reduce_mean(tf.square(mlp_qi))),
                                 name="regularizer")

            train_loss = tf.add(loss_implicit, tf.multiply(q_lambda, regularizer), name="train_loss")

            # optimize
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss, name="optimizer")

            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_implicit, name="optimize")
            # optimizer = tf.train.MomentumOptimizer(0.0001, 0.8).minimize(loss, name="optimize")

            return {
                "user_indices_ph": user_indices,
                "item_indices_ph": item_indices,
                "labels_ph": labels,
                "optimizer": optimizer,
                "train_loss": train_loss,
                "test_loss": loss_implicit,
                "prediction": im_prediction,
                "h_implicit": h_implicit,
                # "global_epoch": global_epoch
            }
