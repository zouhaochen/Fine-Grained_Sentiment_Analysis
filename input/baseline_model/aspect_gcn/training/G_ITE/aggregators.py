import tensorflow as tf

class GCNAggregator_2layer(object):
    def __init__(self, input_dim, output_dim, name, act=tf.nn.leaky_relu):
        self.weight_layer_1 = tf.Variable(
            tf.random_normal([input_dim, int(output_dim/2)]) * tf.sqrt(2 / output_dim),
            name=name+"_1")
        self.weight_layer_2 = tf.Variable(
            tf.random_normal([int(output_dim/2), int(output_dim/2)]) * tf.sqrt(2 / output_dim),
            name=name+"_2")
        self.act = act

    def _call(self, samples):
        next_hidden = []
        self_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[0])
        neig_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[1])
        next_hidden.append(tf.nn.leaky_relu(tf.reduce_mean(tf.concat([neig_1, tf.expand_dims(self_1, axis=1)], axis=1), axis=1)))
        self_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[1])
        neig_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[2])
        next_hidden.append(tf.nn.leaky_relu(tf.reduce_mean(tf.concat([neig_1, tf.expand_dims(self_1, axis=2)], axis=2), axis=2)))
        self_2 = tf.reduce_mean(tf.concat([next_hidden[1], tf.expand_dims(next_hidden[0], axis=1)], axis=1), axis=1)
        self_2 = tf.matmul(self_2, self.weight_layer_2)
        self_2 = self.act(self_2)

        return tf.concat([next_hidden[0], self_2], axis=1)

    def __call__(self, samples):
        outputs = self._call(samples)
        return outputs

    def get_weight(self):
        return [self.weight_layer_1, self.weight_layer_2]


class GCNAggregator_1layer(object):
    def __init__(self, input_dim, output_dim, name, act=tf.nn.leaky_relu):
        self.weight_layer_1 = tf.Variable(
            tf.random_normal([input_dim, output_dim]) * tf.sqrt(2 / output_dim),
            name=name+"_1")
        self.act = act

    def _call(self, samples):
        next_hidden = []
        self_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[0])
        neig_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[1])
        self_1 = tf.reduce_mean(tf.concat([neig_1, tf.expand_dims(self_1, axis=1)], axis=1), axis=1)
        return self.act(self_1)

    def __call__(self, samples):
        outputs = self._call(samples)
        return outputs

    def get_weight(self):
        return [self.weight_layer_1]

    def get_embedding(self, indices):
        return tf.nn.embedding_lookup(self.weight_layer_1, indices)

class GCNAggregator_1layer_weighted(object):
    def __init__(self, input_dim, output_dim, name, act=tf.nn.leaky_relu):
        self.weight_layer_1 = tf.Variable(
            tf.random_normal([input_dim, output_dim]) * tf.sqrt(2 / output_dim),
            name=name+"_1")
        self.act = act

    def _call(self, samples, weights):
        next_hidden = []
        self_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[0])
        neig_1 = tf.nn.embedding_lookup(self.weight_layer_1, samples[1])
        weight = tf.expand_dims(weights[0], axis=2)
        neig_1 = tf.multiply(neig_1, weight)
        self_1 = tf.reduce_sum(tf.concat([neig_1, tf.expand_dims(self_1, axis=1)], axis=1), axis=1)
        return self.act(self_1)

    def __call__(self, samples):
        outputs = self._call(samples)
        return outputs

    def get_weight(self):
        return [self.weight_layer_1]


class MaxPoolingAggregator_1layer(object):
    def __init__(self, input_dim, output_dim, name, act=tf.nn.leaky_relu):
        self.weight_self_layer_1 = tf.Variable(
            tf.random_normal([input_dim, output_dim]) * tf.sqrt(2 / output_dim),
            name=name+"_self_1")
        self.weight_neig_layer_1 = tf.Variable(
            tf.random_normal([2*output_dim, output_dim]) * tf.sqrt(2 / output_dim),
            name=name + "_neig_1")
        self.weight_mlp = tf.Variable(
            tf.random_normal([input_dim, 2*output_dim]) * tf.sqrt(2 / output_dim),
            name=name + "_mlp_1")
        self.bias_mlp = tf.Variable(tf.random_normal([2*output_dim]))
        self.act = act

    def _call(self, samples):
        next_hidden = []
        self_1 = tf.nn.embedding_lookup(self.weight_self_layer_1, samples[0])
        neig_1 = tf.nn.embedding_lookup(self.weight_mlp, samples[1])
        neig_1 = tf.add(neig_1, self.bias_mlp)
        neig_1 = tf.nn.leaky_relu(neig_1)
        neig_1 = tf.reduce_max(neig_1, axis=1)
        neig_1 = tf.matmul(neig_1, self.weight_neig_layer_1)
        self_1 = tf.add_n([self_1, neig_1])
        return self.act(self_1)

    def __call__(self, samples):
        outputs = self._call(samples)
        return outputs

    def get_weight(self):
        return [self.weight_self_layer_1, self.weight_neig_layer_1, self.weight_mlp]




