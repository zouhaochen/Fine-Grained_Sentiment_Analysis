import torch


TRAIN_FILE_PATH = './output/process/fine_grained_sentiment_analysis.train.csv'
TEST_FILE_PATH = './output/process/fine_grained_sentiment_analysis.test.csv'


BIO_O_ID = 0
BIO_B_ID = 1
BIO_I_ID = 2
BIO_MAP = {'O': BIO_O_ID, 'B-ASP': BIO_B_ID, 'I-ASP': BIO_I_ID}


ENT_SIZE = 3


POLA_O_ID = -1
POLA_MAP = ['Negative', 'Positive', 'Neutral']
POLA_DIM = 3


EMBED_PAD_ID = 0
EMBED_MODEL_NAME = './huggingface/LLM_embedder'
EMBED_DIM = 1024


SRD = 3


BATCH_SIZE = 50
EPOCH = 100
LR = 1e-4


MODEL_DIR = './output/model'


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


EPS = 1e-10
LCF = 'cdw'  # cdw cdm fusion

print(DEVICE)
