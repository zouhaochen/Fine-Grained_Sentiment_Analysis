from utils import *
from model import *
from transformers import BertTokenizer

if __name__ == '__main__':
    model = torch.load(MODEL_DIR + 'model_50.pth', map_location=DEVICE)
    with torch.no_grad():
        text = ""
        tokenizer = BertTokenizer.from_pretrained(EMBED_MODEL_NAME)
        tokens = list(text)
        input_ids = tokenizer.encode(tokens)
        mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0).bool()
        pred_ent_label = model.get_entity(input_ids, mask)

        b_ent_pos, b_ent_pola = get_pola(model, input_ids[0], mask[0], pred_ent_label[0])

        if not b_ent_pos:
            print('\t', 'no result.')
        else:
            pred_pair = []
            for ent_pos, pola in zip(b_ent_pos, torch.argmax(b_ent_pola, dim=1)):
                aspect = text[ent_pos[0] - 1:ent_pos[-1]]
                pred_pair.append({'aspect': aspect, 'sentiment': POLA_MAP[pola], 'position': ent_pos})

            print('\t', text)
            print('\t', pred_pair)