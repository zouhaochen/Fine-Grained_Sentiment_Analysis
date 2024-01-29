import itertools
from GCN.training.G_NCF import model_utils
import argparse

def main():
    use_implicit = True
    if use_implicit:
        model_name = "G_NCF"
    else:
        model_name = "G_NCF_without_use_implicit"
    data_scene = ""

    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=str)
    parser.add_argument('--batch', type=str)
    parser.add_argument('--lr', type=str)
    args = parser.parse_args()

    num_factors = [int(item) for item in args.factor.split(',')]
    batch_sizes = [int(item) for item in args.batch.split(',')]
    learning_rates = [float(item) for item in args.lr.split(',')]
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
            "use_implicit": use_implicit,
            "visible_gpu": visible_gpu,
            "max_deg": 10,
            "num_sample": 5
        }

        # init data
        m = model_utils.Utils(hyper_params, save_eval_result=True, save_model=True)
        for num_factor, batch_size, lr, num_neg, num_epoch in itertools.product(num_factors, batch_sizes,
                                                                                learning_rates, num_negs, num_epochs):
            hyper_params = {
                "num_factor": num_factor,
                "batch_size": batch_size,
                "lr": lr,
                "num_neg": num_neg,
                "num_epoch": num_epoch,
                "verbose": 10,
                "eval_top_k": 10,
                "lambda": 0.005,
                "data_name": data_name,
                "model_name": model_name,
                "use_implicit": use_implicit,
                "visible_gpu": visible_gpu
            }
            m.run(hyper_params, save_eval_result=True, save_model=False)


if __name__ == "__main__":
    main()
