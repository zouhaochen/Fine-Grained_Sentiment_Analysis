from utils import *
import torch.utils.data as data
from model import Model

if __name__ == '__main__':

    model = Model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collate_fn)

    for e in range(EPOCH):

        for b, batch in enumerate(loader):
            input_ids, mask, ent_label, ent_cdm, ent_cdw, pola_label, pairs = batch
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            ent_label = ent_label.to(DEVICE)
            ent_cdm = ent_cdm.to(DEVICE)
            ent_cdw = ent_cdw.to(DEVICE)

            pola_label = pola_label.to(DEVICE)
            pred_ent_label = model.get_entity(input_ids, mask)
            pred_pola = model.get_pola(input_ids, mask, ent_cdm, ent_cdw)
            loss = model.loss_fn(input_ids, ent_label, mask, pred_pola, pola_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 10 == 0:
                print('>> epoch', e, 'batch:', b, 'loss:', loss.item())

            correct_cnt = pred_cnt = gold_cnt = 0
            for i in range(len(input_ids)):
                gold_cnt += len(pairs[i])

                b_ent_pos, b_ent_pola = get_pola(model, input_ids[i], mask[i], pred_ent_label[i])
                if not b_ent_pos:
                    continue

                pred_pair = []
                cnt = 0
                for ent, pola in zip(b_ent_pos, torch.argmax(b_ent_pola, dim=1)):
                    pair_item = (ent, pola.item())
                    pred_pair.append(pair_item)

                    if pair_item in pairs[i]:
                        cnt += 1

                correct_cnt += cnt
                pred_cnt += len(pred_pair)

            precision = round(correct_cnt / (pred_cnt + EPS), 3)
            recall = round(correct_cnt / (gold_cnt + EPS), 3)
            f1_score = round(2 / (1 / (precision + EPS) + 1 / (recall + EPS)), 3)
            print('\tcorrect_cnt:', correct_cnt, 'pred_cnt:', pred_cnt, 'gold_cnt:', gold_cnt)
            print('\tprecision:', precision, 'recall:', recall, 'f1_score:', f1_score)

        if e % 3 == 0:
            torch.save(model, MODEL_DIR + f'model_{e}.pth')