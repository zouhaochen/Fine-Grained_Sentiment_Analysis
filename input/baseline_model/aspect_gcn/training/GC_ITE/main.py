import itertools

import sys
# sys.path.append('code/jounal_ite')
from GCN.training.GC_ITE import model_utils
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=str)
    parser.add_argument('--eta', type=str)
    parser.add_argument('--batch', type=str)
    parser.add_argument('--lr', type=str)
    parser.add_argument('--sample', type=int)
    args = parser.parse_args()

    model_name = "GC_ITE"
    data_scene = "without_implicit_in_train/"
    num_factors = [int(item) for item in args.factor.split(',')]
    etas = [float(item) for item in args.eta.split(',')]
    batch_sizes = [int(item) for item in args.batch.split(',')]
    learning_rates = [float(item) for item in args.lr.split(',')]
    num_negs = [4]
    num_epochs = [100]
    num_sample = int(args.sample)

    available_gpu = ["0", "1"]
    visible_gpu = ",".join(available_gpu[0])
    for data_name in ["hetrec2011-lastfm-2k/sparse_70/"]:
        hyper_params = {
            "verbose": 10,
            "eval_top_k": 10,
            "lambda": 0.005,
            "data_name": data_name,
            "data_scene": data_scene,
            "model_name": model_name,
            "visible_gpu": visible_gpu,
            "max_deg": 10,
            "num_sample": num_sample
        }

        # init data
        m = model_utils.Utils(hyper_params, save_eval_result=True, save_model=False)
        for num_factor, eta, batch_size, lr, num_neg, num_epoch in itertools.product(num_factors, etas, batch_sizes,
                                                                                     learning_rates, num_negs,
                                                                                     num_epochs):
            hyper_params = {
                "num_factor": num_factor,
                "eta": eta,
                "batch_size": batch_size,
                "lr": lr,
                "num_neg": num_neg,
                "num_epoch": num_epoch,
                "verbose": 10,
                "eval_top_k": 10,
                "lambda": 0.005,
                "data_name": data_name,
                "data_scene": data_scene,
                "model_name": model_name,
                "visible_gpu": visible_gpu,
            }
            m.run(hyper_params, save_eval_result=True, save_model=False)

if __name__ == "__main__":

    main()
