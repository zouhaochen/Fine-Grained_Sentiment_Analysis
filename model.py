import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from config import *
from torchcrf import CRF
from transformers.models.bert.modeling_bert import BertAttention, BertPooler

from transformers import logging
logging.set_verbosity_error()

config = BertConfig.from_pretrained(EMBED_MODEL_NAME)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(EMBED_MODEL_NAME)
        self.ent_linear = nn.Linear(EMBED_DIM, ENT_SIZE)
        self.crf = CRF(ENT_SIZE, batch_first=True)
        self.pola_linear2 = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.pola_linear3 = nn.Linear(EMBED_DIM * 3, EMBED_DIM)
        self.pola_linear = nn.Linear(EMBED_DIM, POLA_DIM)
        self.attention = BertAttention(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout()

    def get_text_encoded(self, input_ids, mask):
        return self.bert(input_ids, attention_mask=mask)[0]

    def get_entity_fc(self, text_encoded):
        return self.ent_linear(text_encoded)

    def get_entity_crf(self, entity_fc, mask):
        return self.crf.decode(entity_fc, mask)

    def get_entity(self, input_ids, mask):
        text_encoded = self.get_text_encoded(input_ids, mask)
        entity_fc = self.get_entity_fc(text_encoded)
        pred_ent_label = self.get_entity_crf(entity_fc, mask)
        return pred_ent_label

    def get_pola(self, input_ids, mask, ent_cdm, ent_cdw):
        text_encoded = self.get_text_encoded(input_ids, mask)

        # shape [b, c] -> [b, c, 768]
        ent_cdm_weight = ent_cdm.unsqueeze(-1).repeat(1, 1, EMBED_DIM)
        ent_cdw_weight = ent_cdw.unsqueeze(-1).repeat(1, 1, EMBED_DIM)
        cdm_feature = torch.mul(text_encoded, ent_cdm_weight)
        cdw_feature = torch.mul(text_encoded, ent_cdw_weight)

        if LCF == 'fusion':
            out = torch.cat([text_encoded, cdm_feature, cdw_feature], dim=-1)
            out = self.pola_linear3(out)
        if LCF == 'fusion2':
            out = torch.cat([text_encoded, cdw_feature], dim=-1)
            out = self.pola_linear2(out)
        elif LCF == 'cdw':
            out = cdw_feature

        out = self.attention(out, None)

        out = torch.sigmoid(self.pooler(torch.tanh(out[0])))
        return self.pola_linear(out)

    def ent_loss_fn(self, input_ids, ent_label, mask):
        text_encoded = self.get_text_encoded(input_ids, mask)
        entity_fc = self.get_entity_fc(text_encoded)
        return -self.crf.forward(entity_fc, ent_label, mask, reduction='mean')

    def pola_loss_fn(self, pred_pola, pola_label):
        return F.cross_entropy(pred_pola, pola_label)

    def loss_fn(self, input_ids, ent_label, mask, pred_pola, pola_label):
        return self.ent_loss_fn(input_ids, ent_label, mask) + \
        self.pola_loss_fn(pred_pola, pola_label)


class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        if self.combination_type == "gru":
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        node_repesentations, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        return self.update(node_repesentations, aggregated_messages)


class GCNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights

        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])

        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )

        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )

        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_weights))
        x = x1 + x
        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x2 + x
        x = self.postprocess(x)
        node_embeddings = tf.gather(x, input_node_indices)
        return self.compute_logits(node_embeddings)


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


gcn_model = GCNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
)

gcn_model.summary()


if __name__ == '__main__':
    input_ids = torch.randint(0, 3000, (2, 30)).to(DEVICE)
    mask = torch.ones((2, 30)).bool().to(DEVICE)
    model = Model().to(DEVICE)
    ent_cdm = torch.rand((2, 30)).to(DEVICE)
    ent_cdw = torch.rand((2, 30)).to(DEVICE)
    print(model.get_pola(input_ids, mask, ent_cdm, ent_cdw))