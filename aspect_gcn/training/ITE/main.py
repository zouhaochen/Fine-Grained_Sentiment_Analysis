import itertools

import sys
from GCN.training.ITE import model_utils


def main():
    model_name = "ITE"
    data_scene = ""
    num_factors = [8]
    etas = [0.2]
    batch_sizes = [1024, 2048]
    learning_rates = [0.001, 0.01]
    num_negs = [4]
    num_epochs = [100]

    available_gpu = ["0", "1"]
    visible_gpu = ",".join(available_gpu[0])
    for data_name in ["hetrec2011-lastfm-2k/"]:
        hyper_params = {
            "verbose": 10,
            "eval_top_k": 10,
            "lambda": 0.005,
            "data_name": data_name,
            "data_scene": data_scene,
            "model_name": model_name,
            "visible_gpu": visible_gpu
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
                "visible_gpu": visible_gpu
            }
            m.run(hyper_params, save_eval_result=True, save_model=False)


if __name__ == "__main__":
    main()
