import torch

TRAIN_FILE_PATH = './output/process/atepc.train.csv'
TEST_FILE_PATH = './output/process/atepc.test.csv'

# BIO标记相关配置项
BIO_O_ID = 0
BIO_B_ID = 1
BIO_I_ID = 2
BIO_MAP = {'O': BIO_O_ID, 'B-ASP': BIO_B_ID, 'I-ASP': BIO_I_ID}


# 实体标记个数
ENT_SIZE = 3


# 情感分类，值判断积极中性还是消极，对应三分类
POLA_O_ID = -1
POLA_MAP = ['Negative', 'Positive', 'Neutral']
POLA_DIM = 3


# Transformer Embedding相关配置项
# 填充ID为0
BERT_PAD_ID = 0
BERT_MODEL_NAME = './huggingface/LLM_embedder'
BERT_DIM = 1024


# Semantic-Relative Distance
# 距离实体范围为三个字以内
SRD = 3


BATCH_SIZE = 50
EPOCH = 100
LR = 1e-4


MODEL_DIR = './output/model'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


EPS = 1e-10
LCF = 'cdw'  # cdw cdm fusion


print(DEVICE)
