import csv
import itertools
import os
from typing import List

import matplotlib.pyplot as plt

from src.config import config

root_path = config.RESULT_ROOT_PATH


def hyper_params_sensitive(model_name):
    num_factors = [8]
    etas = [0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
    batch_sizes = [512]
    learning_rates = [0.001]
    num_negs = [4]
    num_epochs = [200]
    data_names = ["retail_rocket/"]
    markers = ["o", "s", "v", "*", "x"]
    linestyle = [None, "-", "--", "-", "--"]
    result_dict = {}


def compare_between_models():
    # num_factors = [8, 16, 32, 64]
    num_factors = [64]
    # etas = [0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
    etas = [1.0]
    # batch_sizes = [128, 256, 512]
    batch_sizes = [512]
    learning_rates = [0.001]
    num_negs = [4]
    num_epochs = [200]
    data_names = ["retail_rocket/"]
    model_names = ["ITE", "ITE_scalable", "ITE_implicit_upstair"]
    markers = ["o", "s", "v", "*", "x"]
    linestyle = [None, "-", "--", "-", "--"]
    result_dict = {}
    for data_name in data_names:
        data_root_path = root_path + data_name
        for num_factor, eta, batch_size, lr, num_neg, num_epoch in itertools.product(num_factors, etas, batch_sizes,
                                                                                     learning_rates, num_negs,
                                                                                     num_epochs):
            for model_name in model_names:
                path = data_root_path + "eval_results/{}/{}/{}_{}_{}_{}_{}".format(model_name,
                                                                                   num_epoch,
                                                                                   num_factor,
                                                                                   eta,
                                                                                   batch_size,
                                                                                   lr,
                                                                                   num_neg)
                if os.path.isfile(path):
                    epoch_list = []
                    hit_list = []
                    ndcg_list = []
                    with open(path) as file:
                        line = next(file)
                        while "init" not in line:
                            line = next(file)
                        line = next(file)
                        while not line.startswith("+-------"):
                            epoch = int(line.split("|")[1].strip())
                            hit = float(line.split("|")[-3].strip()) * 100
                            ndcg = float(line.split("|")[-2].strip()) * 100

                            epoch_list.append(epoch)
                            hit_list.append(hit)
                            ndcg_list.append(ndcg)
                            line = next(file)
                    if (eta, batch_size) not in result_dict:
                        result_dict[(eta, batch_size)] = {}
                    result_dict[(eta, batch_size)][model_name] = [epoch_list, hit_list, ndcg_list]

        # print(result_dict)
        for params in result_dict:
            axs: List[plt.Axes]
            fig, axs = plt.subplots(nrows=2, ncols=1, num=None, figsize=(6, 9), dpi=120,
                                    facecolor="w",
                                    edgecolor="k")
            for i, model_name in enumerate(result_dict[params]):
                fig.suptitle("Factor: {}, Eta: {}, Batchsize: {}, lr: {}".format(num_factors[0], params[0], params[1],
                                                                                 learning_rates[0]), fontsize=16)
                res = result_dict[params][model_name]
                epoch_list = res[0]
                hit_list = res[1]
                ndcg_list = res[2]

                # --------------------- HIT ----------------------
                leg, = axs[0].plot(epoch_list, hit_list, marker=markers[i],
                                   label=model_names[i],
                                   linewidth=1.5, markersize=4.0, linestyle=linestyle[i])

                axs[0].set_ylabel("HR@10 (%)", fontsize=16)
                for axis in ["top", "bottom", "left", "right"]:
                    axs[0].spines[axis].set_linewidth(2)
                axs[0].xaxis.set_major_locator(plt.MaxNLocator(6))
                axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))
                # axs[0].axis("tight")
                axs[0].legend()

                # --------------------- NDCG ----------------------
                leg, = axs[1].plot(epoch_list, ndcg_list, marker=markers[i],
                                   label=model_names[i],
                                   linewidth=1.5, markersize=4.0, linestyle=linestyle[i])
                axs[1].set_ylabel("NDCG@10 (%)", fontsize=16)
                for axis in ["top", "bottom", "left", "right"]:
                    axs[1].spines[axis].set_linewidth(2)
                axs[1].xaxis.set_major_locator(plt.MaxNLocator(6))
                axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))
                # axs[1].axis("tight")
                axs[1].legend()

            plt.subplots_adjust(top=0.87,
                                bottom=0.1,
                                left=0.21,
                                right=0.765,
                                hspace=0.43,
                                wspace=0.185)
    plt.show()


def compare_between_models_with_num_factor():
    num_factors = [8, 16, 32, 64]
    batch_sizes = [2048] * 5
    etas = [0.5, 0.5, 0.5, 1.0, 0.5]
    model_names = ["one_hot_log_loss", "item_pcat_log_loss", "both_concat_onehot", "NMTR", "mtmf_model"]
    label_names = ["ITE_0", "ITE_1", "ITE_2", "NMTR", "MTMF"]
    data_names = ["recobell", "retail_rocket"]
    marker_list = ["o", "s", "v", "*", "x"]
    linestyle = [None, "-", "--", "-", "--"]
    # , "ITE_item_pcat" "item_pcat_log_loss", "." , 0.5
    # "recobell",
    # data = "lotte"
    # # data = "movielens-1m"
    # # data = "movielens-100k"
    # # data = "recobell"
    # # data = "retailrocket"
    # # data = "yes24"

    for data in data_names:
        hit_results = []
        ndcg_results = []
        for z in range(len(model_names)):
            model = model_names[z]
            hit_res = []
            ndcg_res = []
            for factor in num_factors:

                path = root_path + data + "/" + model + "/num_factor/{}_{}_{}".format(factor, batch_sizes[z],
                                                                                      etas[z])
                # print(path)
                try:
                    with open(path) as file:
                        for line in file:
                            if "| 50    |" in line:
                                hit = float(line.split("|")[-3].strip())
                                ndcg = float(line.split("|")[-2].strip())
                                hit_res.append(hit)
                                ndcg_res.append(ndcg)
                except FileNotFoundError:
                    print("?")
            hit_results.append(hit_res)
            ndcg_results.append(ndcg_res)
        for re in ndcg_results:
            print(re)

        # --------------------- HIT ----------------------
        plt.figure(num=None, figsize=(16, 9), dpi=120, facecolor="w", edgecolor="k")
        plt.subplot(121)
        for z in range(len(model_names)):
            plt.plot(num_factors, hit_results[z], marker=marker_list[z], label=label_names[z], linewidth=1.5,
                     markersize=5.0, linestyle=linestyle[z])

        plt.ylabel("HR@10", fontweight="semibold")
        plt.xlabel("Factors", fontweight="semibold")
        plt.title("Dataset: {}".format(data), fontweight="semibold")

        plt.axis("tight")
        plt.legend(fontsize=9)

        # plt.grid()
        # plt.xticks(np.arange(min(num_factor), max(num_factor) + 8, 8))
        plt.xticks(num_factors)
        # --------------------- NDCG -----------------------
        # plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor="w", edgecolor="k")
        plt.subplot(122)
        for z in range(len(model_names)):
            plt.plot(num_factors, ndcg_results[z], marker=marker_list[z], label=label_names[z], linewidth=1.5,
                     markersize=5.0, linestyle=linestyle[z])
        # plt.grid()

        plt.ylabel("NDCG@10", fontweight="semibold")
        plt.xlabel("Factors", fontweight="semibold")
        plt.title("Dataset: {}".format(data), fontweight="semibold")

        plt.axis("tight")
        plt.legend(fontsize=9)

        # plt.grid()
        plt.xticks(num_factors)

        # plt.subplots_adjust(top=0.94,
        #                     bottom=0.1,
        #                     left=0.285,
        #                     right=0.695,
        #                     hspace=0.365,
        #                     wspace=0.215)
    plt.show()


def compare_between_models_with_num_factor_2():
    num_factors = [8, 16, 32, 64]
    batch_sizes = [2048] * 5
    etas = [0.5, 0.5, 0.5, 1.0, 0.5]
    model_names = ["one_hot_log_loss", "item_pcat_log_loss", "both_concat_onehot", "NMTR", "mtmf_model"]
    label_names = ["ITE", "ITE-OSSi", "ITE-Si", "NMTR", "MTMF"]
    data_names = ["recobell", "retail_rocket"]
    data_title_names = ["Recobell", "Retailrocket"]
    marker_list = ["o", "s", "v", "d", "x"]
    linestyle = [None, "-", "--", "-", "--"]
    # , "ITE_item_pcat" "item_pcat_log_loss", "." , 0.5
    # "recobell",
    # data = "lotte"
    # # data = "movielens-1m"
    # # data = "movielens-100k"
    # # data = "recobell"
    # # data = "retailrocket"
    # # data = "yes24"

    for d in range(len(data_names)):
        lengend_names = []
        legend_handlers = []
        hit_results = []
        ndcg_results = []
        for z in range(len(model_names)):
            model = model_names[z]
            hit_res = []
            ndcg_res = []
            for factor in num_factors:

                path = root_path + data_names[d] + "/" + model + "/num_factor/{}_{}_{}".format(factor, batch_sizes[z],
                                                                                               etas[z])
                # print(path)
                try:
                    with open(path) as file:
                        for line in file:
                            if "| 50    |" in line:
                                hit = float(line.split("|")[-3].strip()) * 100
                                ndcg = float(line.split("|")[-2].strip()) * 100
                                hit_res.append(hit)
                                ndcg_res.append(ndcg)
                except FileNotFoundError:
                    print("?")
            hit_results.append(hit_res)
            ndcg_results.append(ndcg_res)
        for re in ndcg_results:
            print(re)

        # --------------------- HIT ----------------------
        ax1: plt.Axes
        ax2: plt.Axes
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num=None, figsize=(16, 9), dpi=120,
                                       facecolor="w",
                                       edgecolor="k")
        # fig.suptitle("{}".format(data_title_names[d]), fontsize=24)
        for z in range(len(model_names)):
            leg, = ax1.plot(num_factors, hit_results[z], marker=marker_list[z], label=label_names[z], linewidth=2.0,
                            markersize=5.5, linestyle=linestyle[z])
            legend_handlers.append(leg)
            lengend_names.append(label_names[z])
        # ax1.locator_params(nbins=3)
        ax1.set_xlabel("Factor", fontsize=16)
        ax1.set_ylabel("HR@10 (%)", fontsize=16)
        # ax1.set_title("Dataset: {}".format(data), fontsize=24, fontweight="semibold")
        ax1.set_xticks(num_factors)
        # ax1.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
        # ax1.legend(prop={"size": 30}, loc="lower center", shadow=True, bbox_to_anchor=(1.2, -0.5))
        plt.axis("tight")
        for axis in ["top", "bottom", "left", "right"]:
            ax1.spines[axis].set_linewidth(2)

        # --------------------- NDCG ----------------------
        # ax1: plt.Axes
        # ax2: plt.Axes
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num=None, figsize=(16, 9), dpi=120, facecolor="w",
        #                                edgecolor="k")

        for z in range(len(model_names)):
            ax2.plot(num_factors, ndcg_results[z], marker=marker_list[z], label=label_names[z], linewidth=2.0,
                     markersize=5.5, linestyle=linestyle[z])
        # ax2.locator_params(nbins=3)
        ax2.set_xlabel("Factor", fontsize=16)
        ax2.set_ylabel("NDCG@10 (%)", fontsize=16)
        # ax2.set_title("Dataset: {}".format(data), fontsize=24, fontweight="semibold")
        ax2.set_xticks(num_factors)
        ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
        # plt.legend(bbox_to_anchor=(1.2, 0.5), loc="cennter left", bbox_transform=plt.gcf().transFigure)
        # plt.tight_layout()
        plt.axis("tight")
        for axis in ["top", "bottom", "left", "right"]:
            ax2.spines[axis].set_linewidth(2)

        plt.legend(legend_handlers, lengend_names, prop={"size": 20}, loc="lower center", shadow=True, ncol=6,
                   bbox_to_anchor=(-0.12, -0.3))

        # plt.figure(num=None, figsize=(16, 9), dpi=120, facecolor="w", edgecolor="k")
        # plt.subplot(121)
        # for z in range(len(model_names)):
        #     plt.plot(num_factor, hit_results[z], marker=marker_list[z], label=label_names[z], linewidth=1.5,
        #              markersize=5.0, linestyle=linestyle[z])
        #
        # plt.ylabel("HR@10", fontweight="semibold")
        # plt.xlabel("Factors", fontweight="semibold")
        # plt.title("Dataset: {}".format(data), fontweight="semibold")
        #
        # plt.axis("tight")
        # plt.legend(fontsize=9)
        #
        # # plt.grid()
        # # plt.xticks(np.arange(min(num_factor), max(num_factor) + 8, 8))
        # plt.xticks(num_factor)
        # # --------------------- NDCG -----------------------
        # # plt.figure(num=None, figsize=(10, 10), dpi=120, facecolor="w", edgecolor="k")
        # plt.subplot(122)
        # for z in range(len(model_names)):
        #     plt.plot(num_factor, ndcg_results[z], marker=marker_list[z], label=label_names[z], linewidth=1.5,
        #              markersize=5.0, linestyle=linestyle[z])
        # # plt.grid()
        #
        # plt.ylabel("NDCG@10", fontweight="semibold")
        # plt.xlabel("Factors", fontweight="semibold")
        # plt.title("Dataset: {}".format(data), fontweight="semibold")
        #
        # plt.axis("tight")
        # plt.legend(fontsize=9)
        #
        # # plt.grid()
        # plt.xticks(num_factor)
        #
        plt.subplots_adjust(top=0.93,
                            bottom=0.245,
                            left=0.075,
                            right=0.95,
                            hspace=0.205,
                            wspace=0.3)
    plt.show()


def factor_through_epochs():
    num_factors = [8, 16, 32, 64]
    batch_sizes = [2048] * 5
    etas = [0.5, 0.5, 0.5, 1.0, 0.5]
    model_names = ["one_hot_log_loss", "item_pcat_log_loss", "both_concat_onehot", "NMTR", "mtmf_model"]
    label_names = ["ITE", "ITE-OSSi", "ITE-Si", "NMTR", "MTMF"]
    data_names = ["recobell", "retail_rocket"]
    data_title_names = ["Recobell", "Retailrocket"]
    marker_list = ["o", "s", "v", "d", "x"]
    linestyle = [None, "-", "--", "-", "--"]
    # x_tick = [10, 20, 30, 40, 50]
    # y_ticks = {data_names[0]: {"hit": [60, 70, 80, 85], "ndcg": [25, 40, 50, 60, 70]},
    #            data_names[1]: {"hit": [15, 30, 45, 60], "ndcg": [0, 15, 30, 45]}}
    # "recobell",
    # data = "lotte"
    # # data = "movielens-1m"
    # # data = "movielens-100k"
    # # data = "recobell"
    # # data = "retailrocket"
    # # data = "yes24"

    for d in range(len(data_names)):
        lengend_names = []
        legend_handlers = []
        model_epochs = []
        hit_results = []
        ndcg_results = []
        for i in range(len(num_factors)):
            factor_model_e = []
            factor_hit = []
            factor_ndcg = []
            for z in range(len(model_names)):
                model_e = []
                hit_res = []
                ndcg_res = []
                model = model_names[z]
                path = root_path + data_names[d] + "/" + model + "/num_factor/{}_{}_{}".format(num_factors[i],
                                                                                               batch_sizes[z],
                                                                                               etas[z])
                # print(path)
                try:
                    with open(path) as file:
                        for line in file:
                            if "init" not in line:
                                continue
                            else:
                                break
                        next(file)
                        for line in file:
                            # print(line)
                            if not line.startswith("+-------"):
                                e = int(line.split("|")[1].strip())
                                # print(e)
                                hit = float(line.split("|")[-3].strip()) * 100
                                ndcg = float(line.split("|")[-2].strip()) * 100
                                model_e.append(e)
                                hit_res.append(hit)
                                ndcg_res.append(ndcg)
                            else:
                                break
                except FileNotFoundError:
                    print("?")
                factor_model_e.append(model_e)
                factor_hit.append(hit_res)
                factor_ndcg.append(ndcg_res)
            hit_results.append(factor_hit)
            ndcg_results.append(factor_ndcg)
            model_epochs.append(factor_model_e)
        # for re in ndcg_results:
        #     print(re)
        axs: List[List[plt.Axes]]
        fig, axs = plt.subplots(nrows=2, ncols=4, num=None, figsize=(16, 9), dpi=120,
                                facecolor="w",
                                edgecolor="k")
        # fig.suptitle("{}".format(data_title_names[d]), fontsize=24)
        # --------------------- HIT ----------------------

        for i in range(len(num_factors)):

            # plt.subplot(2, 4, i + 1)
            for z in range(len(model_names)):
                leg, = axs[0][i].plot(model_epochs[i][z], hit_results[i][z], marker=marker_list[z],
                                      label=label_names[z],
                                      linewidth=1.5, markersize=4.0, linestyle=linestyle[z])

                if i == 0:
                    legend_handlers.append(leg)
                    lengend_names.append(label_names[z])
                    axs[0][i].set_ylabel("HR@10 (%)", fontsize=16)
            # ax[0][i].set_xlabel("Epoch", fontsize=14, fontweight="semibold")
            axs[0][i].set_title("Factors: {}".format(num_factors[i]), fontsize=16)
            for axis in ["top", "bottom", "left", "right"]:
                axs[0][i].spines[axis].set_linewidth(2)
            # axs[0][i].set_xticks(x_tick)
            # axs[0][i].set_yticks(y_ticks[data]["hit"])
            axs[0][i].xaxis.set_major_locator(plt.MaxNLocator(6))
            axs[0][i].yaxis.set_major_locator(plt.MaxNLocator(4))
            plt.axis("tight")
            # plt.legend(fontsize=9)
        # \n(Learning rate: 0.001; Batch size: 2048; Num factors: {})".format(num_factor[i])
        for i in range(len(num_factors)):
            # plt.subplot(2, 4, i + 5)
            for z in range(len(model_names)):
                leg, = axs[1][i].plot(model_epochs[i][z], ndcg_results[i][z], marker=marker_list[z],
                                      label=label_names[z],
                                      linewidth=1.5, markersize=4.0, linestyle=linestyle[z])

            if i == 0:
                axs[1][i].set_ylabel("NDCG@10 (%)", fontsize=16)
            axs[1][i].set_xlabel("Epoch", fontsize=16)
            # axs[1][i].set_title("Factors: {}".format(num_factor[i]), fontsize=14, fontweight="semibold")
            for axis in ["top", "bottom", "left", "right"]:
                axs[1][i].spines[axis].set_linewidth(2)
            # axs[1][i].set_xticks(x_tick)
            # axs[1][i].set_yticks(y_ticks[data]["ndcg"])
            axs[1][i].xaxis.set_major_locator(plt.MaxNLocator(6))
            axs[1][i].yaxis.set_major_locator(plt.MaxNLocator(4))
            plt.axis("tight")
        plt.legend(legend_handlers, lengend_names, prop={"size": 20}, loc="lower center", shadow=True, ncol=6,
                   bbox_to_anchor=(-1.35, -0.7))
        # plt.legend(fontsize=9)
        # \n(Learning rate: 0.001; Batch size: 2048; Num factors: {})".format(num_factor[i])
        # plt.grid()
        # plt.xticks(np.arange(min(num_factor), max(num_factor) + 8, 8))
        # plt.xticks(num_factor)
        # --------------------- NDCG -----------------------
        # plt.subplot(2, 1, 2)
        # for z in range(len(model_names)):
        #     plt.plot(model_epochs[z], ndcg_results[z], marker=marker_list[z], label=label_names[z])
        # # plt.grid()
        #
        # plt.ylabel("NDCG@10")
        # plt.xlabel("Number factors \n(Learning rate: 0.001; Batch size: 2048; Epochs: 51)")
        # plt.title("Data: {}".format(data))
        #
        # plt.legend()
        # plt.grid()
        # plt.xticks(num_factor)

        plt.subplots_adjust(top=0.95,
                            bottom=0.215,
                            left=0.05,
                            right=0.99,
                            hspace=0.325,
                            wspace=0.225)
    plt.show()


def eta_through_epoch():
    num_factors = [8] * 2
    batch_sizes = [2048] * 2
    etas = [0.1, 0.5, 1.0, 2.0]
    model_names = ["one_hot_log_loss", "mtmf_model"]
    label_names = ["0.1", "0.5", "1.0", "2.0"]
    data_names = ["recobell", "retail_rocket"]
    marker_list = ["o", "s", "v", "*", "x"]
    linestyle = [None, "-", "--", "-", "--"]
    # "recobell",
    # data = "lotte"
    # # data = "movielens-1m"
    # # data = "movielens-100k"
    # # data = "recobell"
    # # data = "retailrocket"
    # # data = "yes24"

    # begin for each dataset
    for data in data_names:
        model_epochs = []
        hit_results = []
        ndcg_results = []
        # begin for each model
        for i in range(len(model_names)):
            model_name = model_names[i]
            eta_model_e = []
            eta_hit = []
            eta_ndcg = []
            # begin for each eta
            for j in range(len(etas)):
                eta = etas[j]
                model_e = []
                hit_res = []
                ndcg_res = []

                path = root_path + data + "/" + model_name + "/eta/{}_{}_{}".format(num_factors[i], batch_sizes[i],
                                                                                    eta)
                # print(path)
                # begin try-except
                try:
                    with open(path) as file:
                        for line in file:
                            if "init" not in line:
                                continue
                            else:
                                break
                        next(file)
                        for line in file:
                            # print(line)
                            if not line.startswith("+-------"):
                                e = int(line.split("|")[1].strip())
                                # print(e)
                                hit = float(line.split("|")[-3].strip())
                                ndcg = float(line.split("|")[-2].strip())
                                model_e.append(e)
                                hit_res.append(hit)
                                ndcg_res.append(ndcg)
                            else:
                                break
                except FileNotFoundError:
                    print("File not found???")
                # end try-except
                eta_model_e.append(model_e)
                eta_hit.append(hit_res)
                eta_ndcg.append(ndcg_res)
            # end for each eta
            model_epochs.append(eta_model_e)
            hit_results.append(eta_hit)
            ndcg_results.append(eta_ndcg)

        # end for each model

        # for re in ndcg_results:
        #     print(re)

        # --------------------- HIT ----------------------
        figs = plt.figure(num=data, figsize=(18, 9), facecolor="w", edgecolor="k")
        for i in range(len(model_names)):
            plt.subplot(2, 2, i + 1)
            for j in range(len(etas)):
                plt.plot(model_epochs[i][j], hit_results[i][j], marker=marker_list[j], label=label_names[j],
                         linewidth=2.0, markersize=6.0, linestyle=linestyle[j])

            plt.ylabel("HR@10", fontweight="semibold")
            plt.xlabel("Epoch", fontweight="semibold")
            plt.title("Model: {}".format(model_names[i]), fontweight="semibold")

            plt.axis("tight")
            plt.legend(fontsize=9)
        # \n(Learning rate: 0.001; Batch size: 2048; Num factors: {})".format(num_factor[i])
        for i in range(len(model_names)):
            plt.subplot(2, 2, i + 3)
            for j in range(len(etas)):
                plt.plot(model_epochs[i][j], ndcg_results[i][j], marker=marker_list[j], label=label_names[j],
                         linewidth=2.0, markersize=6.0, linestyle=linestyle[j])

            plt.ylabel("NDCG@10", fontweight="semibold")
            plt.xlabel("Epoch", fontweight="semibold")
            plt.title("Model: {}".format(model_names[i]), fontweight="semibold")

            plt.axis("tight")
            plt.legend(fontsize=9)
        # \n(Learning rate: 0.001; Batch size: 2048; Num factors: {})".format(num_factor[i])
        # plt.grid()
        # plt.xticks(np.arange(min(num_factor), max(num_factor) + 8, 8))
        # plt.xticks(num_factor)
        # --------------------- NDCG -----------------------
        # plt.subplot(2, 1, 2)
        # for z in range(len(model_names)):
        #     plt.plot(model_epochs[z], ndcg_results[z], marker=marker_list[z], label=label_names[z])
        # # plt.grid()
        #
        # plt.ylabel("NDCG@10")
        # plt.xlabel("Number factors \n(Learning rate: 0.001; Batch size: 2048; Epochs: 51)")
        # plt.title("Data: {}".format(data))
        #
        # plt.legend()
        # plt.grid()
        # plt.xticks(num_factor)

        # plt.subplots_adjust(top=0.956,
        #                     bottom=0.071,
        #                     left=0.047,
        #                     right=0.989,
        #                     hspace=0.243,
        #                     wspace=0.247)
    plt.show()


def compare_ite_vcc():
    # , "both_concat_embed_added_zone" "ITE-3",
    model_names = ["both_concat_embed", "both_concat_embed_added_zone", "both_concat_embed_added_zone_and_doc"]
    label_names = ["ITE", "ITE-zone", "Con-ITE"]
    num_factor = 32
    batch_size = 1024
    eta = 0.5
    data_names = "vccorp"
    marker_list = ["o", "s", "v"]
    list_epoch_18_total = []
    hit_18_total = []
    recall_18_total = []
    list_epoch_19_total = []
    hit_19_total = []
    recall_19_total = []
    for z in range(len(model_names)):
        list_epoch_18 = []
        hit_18 = []
        recall_18 = []
        list_epoch_19 = []
        hit_19 = []
        recall_19 = []
        model = model_names[z]
        path = root_path + data_names + "/log/" + model + "/batch_size/1024.log"
        result = []
        with open(path) as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar="|", quoting=csv.QUOTE_ALL)
            for row in csv_reader:
                result.append(row)

        for row in result[10:25]:
            e = int(row[1])
            hit = float(row[3])
            recall = float(row[4])
            list_epoch_18.append(e)
            hit_18.append(hit)
            recall_18.append(recall)
        list_epoch_18_total.append(list_epoch_18)
        hit_18_total.append(hit_18)
        recall_18_total.append(recall_18)
        for row in result[26:]:
            e = int(row[1])
            hit = float(row[3])
            recall = float(row[4])
            list_epoch_19.append(e)
            hit_19.append(hit)
            recall_19.append(recall)
        list_epoch_19_total.append(list_epoch_19)
        hit_19_total.append(hit_19)
        recall_19_total.append(recall_19)

    plt.figure(num=None, figsize=(18, 9), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(121)
    for z in range(len(model_names)):
        plt.plot(list_epoch_18_total[z], hit_18_total[z], marker=marker_list[z], label=label_names[z],
                 linewidth=1.5,
                 markersize=5.0)

    plt.ylabel("HR@10", fontweight="semibold")
    plt.xlabel("Epoch", fontweight="semibold")
    plt.title("Date: 2019/04/18", fontweight="semibold")

    plt.axis("tight")
    plt.legend(fontsize=9)

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    # plt.figure(num=None, figsize=(18, 9), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(122)
    for z in range(len(model_names)):
        plt.plot(list_epoch_18_total[z], recall_18_total[z], marker=marker_list[z], label=label_names[z],
                 linewidth=1.5,
                 markersize=5.0)

    plt.ylabel("RECALL@10", fontweight="semibold")
    plt.xlabel("Epoch", fontweight="semibold")
    plt.title("Date: 2019/04/18", fontweight="semibold")

    plt.axis("tight")
    plt.legend(fontsize=9)

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    plt.figure(num=None, figsize=(18, 9), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(121)
    for z in range(len(model_names)):
        plt.plot(list_epoch_19_total[z], hit_19_total[z], marker=marker_list[z], label=label_names[z],
                 linewidth=1.5,
                 markersize=5.0)

    plt.ylabel("HR@10", fontweight="semibold")
    plt.xlabel("Epoch", fontweight="semibold")
    plt.title("Date: 2019/04/19", fontweight="semibold")

    plt.axis("tight")
    plt.legend(fontsize=9)

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    # plt.figure(num=None, figsize=(18, 9), dpi=120, facecolor="w", edgecolor="k")
    plt.subplot(122)
    for z in range(len(model_names)):
        plt.plot(list_epoch_19_total[z], recall_19_total[z], marker=marker_list[z], label=label_names[z],
                 linewidth=1.5,
                 markersize=5.0)

    plt.ylabel("RECALL@10", fontweight="semibold")
    plt.xlabel("Epoch", fontweight="semibold")
    plt.title("Date: 2019/04/19", fontweight="semibold")

    plt.axis("tight")
    plt.legend(fontsize=9)

    plt.xticks(list_epoch_18_total[0][:-1] + [70])
    plt.show()


def main():
    # font = {"weight": "bold",
    #         "size": 16}
    # matplotlib.rc("font", **font)

    plt.rcParams.update({
        "font.family": "sans-serif"})
    plt.rcParams.update({
        "font.size": 18,
        # "font.weight": "semibold",
        "font.sans-serif": "Verdana"})
    compare_between_models()
    # compare_between_models_with_num_factor_2()
    # compare_ite_vcc()
    # eta_through_epoch()


if __name__ == "__main__":
    main()
