import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Union, Dict, Any
import adabound
from model import regularize
from model import model1 as ml
from model import calc


# == CONFIG ==================================================================================
def get_hyperParam():
    return {
        "cross_num": 4,
        "seed": 2015,
        "criterion": "CrossEntropyLoss",
        "optimizer": "Adam",
        "batch_size": 5,
        "epoch_num": 5000,
        "learning_rate": 0.0001,
        "amsgrad_bool": True,  # -- for Adam optimizer
        "isSave": True
    }


def set_criterion(loss_type: str) -> Any:
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        print("ERROR in set_criterion")
        sys.exit()


def set_optimizer(optim_type: str, net_params: Any, lr: float, amsgrad_bool: bool = True) -> Any:
    if optim_type == "Adam":
        return optim.Adam(net_params, lr=lr, amsgrad=amsgrad_bool)
    elif optim_type == "adabound":
        return adabound.AdaBound(net_params, lr=lr)
    else:
        print("ERROR in set_optimizer")
        sys.exit()


# =============================================================================================
def main1():
    # -- hyper params setting
    hp = get_hyperParam()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- data setting
    reg = regularize.Regularize1(train_data_path_base=r"/train.csv",
                                 test_data_path_base=r"/test.csv")
    train = reg.read_origin_data("train")
    test = reg.read_origin_data("test")
    train = reg.regularize(train)
    test = reg.regularize(test)
    train_split = reg.split_in_cat(train, cat_col_num=11)
    test_ready = reg.set_input_data(torch.Tensor(test.values), device, "double")

    # -- start learning
    for k in range(hp["cross_num"]):
        # for k in range(1):
        # -- network
        net = ml.Net()
        net.to(device)
        net.double()
        print("device: " + str(device))

        criterion = set_criterion(hp["criterion"])
        optimizer = set_optimizer(hp["optimizer"], net_params=net.parameters(), lr=hp["learning_rate"],
                                  amsgrad_bool=hp["amsgrad_bool"])

        # -- load param if necessary
        # net.load_state_dict(torch.load(param_path, map_location=device))

        # -- start train
        net.train()
        tr_val_0 = reg.split_valid_train(torch.Tensor(train_split[0].values), hp["cross_num"], k, hp["seed"])
        tr_val_1 = reg.split_valid_train(torch.Tensor(train_split[1].values), hp["cross_num"], k, hp["seed"])
        tr_val_2 = reg.split_valid_train(torch.Tensor(train_split[2].values), hp["cross_num"], k, hp["seed"])
        tr_val_3 = reg.split_valid_train(torch.Tensor(train_split[3].values), hp["cross_num"], k, hp["seed"])

        best_score0 = [[0.0, ""], [0.0, ""], [0.0, ""]]  # -- for acc_total
        best_score1 = [[0.0, ""], [0.0, ""], [0.0, ""]]  # -- for acc_ave
        best_score2 = [[0.0, ""], [0.0, ""], [0.0, ""]]  # -- for kappa
        isbest = False
        isdiversity = True
        b_size = hp["batch_size"]
        for epoch in range(hp["epoch_num"]):
            # -- training
            net.train()
            dl0 = list(DataLoader(tr_val_0[0], batch_size=b_size, shuffle=True))
            dl1 = list(DataLoader(tr_val_1[0], batch_size=b_size, shuffle=True))
            dl2 = list(DataLoader(tr_val_2[0], batch_size=b_size, shuffle=True))
            dl3 = list(DataLoader(tr_val_3[0], batch_size=b_size, shuffle=True))

            for i in range(min(len(dl0), len(dl1), len(dl2), len(dl3))):
                data = torch.cat((dl0[i], dl1[i], dl2[i], dl3[i]), dim=0)
                d_ready = reg.set_input_data(data, device=device, dtype="double")

                optimizer.zero_grad()
                out = net(d_ready[0])
                loss = criterion(out, d_ready[1])
                loss.backward()
                optimizer.step()

            # -- evaluate
            net.eval()
            vdl0 = list(DataLoader(tr_val_0[1], batch_size=min([len(tr_val_0[1]),
                                                                len(tr_val_1[1]),
                                                                len(tr_val_2[1]),
                                                                len(tr_val_3[1])]), shuffle=True))
            vdl1 = list(DataLoader(tr_val_1[1], batch_size=min([len(tr_val_0[1]),
                                                                len(tr_val_1[1]),
                                                                len(tr_val_2[1]),
                                                                len(tr_val_3[1])]), shuffle=True))
            vdl2 = list(DataLoader(tr_val_2[1], batch_size=min([len(tr_val_0[1]),
                                                                len(tr_val_1[1]),
                                                                len(tr_val_2[1]),
                                                                len(tr_val_3[1])]), shuffle=True))
            vdl3 = list(DataLoader(tr_val_3[1], batch_size=min([len(tr_val_0[1]),
                                                                len(tr_val_1[1]),
                                                                len(tr_val_2[1]),
                                                                len(tr_val_3[1])]), shuffle=True))
            val = torch.cat((vdl0[0], vdl1[0], vdl2[0], vdl3[0]), dim=0)
            vd_ready = reg.set_input_data(val, device=device, dtype="double")
            with torch.no_grad():
                # -- validation
                out = net(vd_ready[0])
                # ev = calc.EvaluateCat(out, torch.squeeze(vd_ready[1]).long())
                threshold_optim = calc.NelderMead(out, torch.squeeze(vd_ready[1]).long())
                threshold = threshold_optim.optimize()
                ev = calc.EvaluateCatWithThreshold(out, torch.squeeze(vd_ready[1]).long(), threshold)

                # -- save condition check
                if ev.diversity() < 0.8:
                    isdiversity = False

                if hp["isSave"]:
                    if best_score0[-1][0] <= ev.acc_total() and ev.diversity() > 0.9:
                        isbest = True
                        if os.path.exists(r"./save/save_param" + r"/net" + best_score0[-1][1] + r'.ckpt'):
                            os.remove(r"./save/save_param" + r"/net" + best_score0[-1][1] + r'.ckpt')
                        if os.path.exists(r"./save/save_param" + r"/threshold" + best_score0[-1][1] + r'.csv'):
                            os.remove(r"./save/save_param" + r"/threshold" + best_score0[-1][1] + r'.csv')
                        id = r"0_" + str(k) + r"_" + str(epoch)
                        best_score0 = best_score0[:2]
                        best_score0 += [[ev.acc_total(), id]]
                        best_score0.sort(reverse=True)
                        print(best_score0)
                        torch.save(net.state_dict(), r"./save/save_param" + r"/net" + id + r'.ckpt')
                        # pd.DataFrame(threshold.numpy()).to_csv(r"./save/save_param" + r"/threshold" + id + r'.csv', index=None)

                    if best_score1[-1][0] <= ev.acc_ave() and ev.diversity() > 0.9:
                        isbest = True
                        if os.path.exists(r"./save/save_param" + r"/net" + best_score1[-1][1] + r'.ckpt'):
                            os.remove(r"./save/save_param" + r"/net" + best_score1[-1][1] + r'.ckpt')
                        if os.path.exists(r"./save/save_param" + r"/threshold" + best_score1[-1][1] + r'.csv'):
                            os.remove(r"./save/save_param" + r"/threshold" + best_score1[-1][1] + r'.csv')
                        id = r"1_" + str(k) + r"_" + str(epoch)
                        best_score1 = best_score1[:2]
                        best_score1 += [[ev.acc_ave(), id]]
                        best_score1.sort(reverse=True)
                        print(best_score1)
                        torch.save(net.state_dict(), r"./save/save_param" + r"/net" + id + r'.ckpt')
                        # pd.DataFrame(threshold.numpy()).to_csv(r"./save/save_param" + r"/threshold" + id + r'.csv', index=None)

                    if best_score2[-1][0] <= ev.quadratic_weighted_kappa() and ev.diversity() > 0.9:
                        isbest = True
                        if os.path.exists(r"./save/save_param" + r"/net" + best_score2[-1][1] + r'.ckpt'):
                            os.remove(r"./save/save_param" + r"/net" + best_score2[-1][1] + r'.ckpt')
                        if os.path.exists(r"./save/save_param" + r"/threshold" + best_score2[-1][1] + r'.csv'):
                            os.remove(r"./save/save_param" + r"/threshold" + best_score2[-1][1] + r'.csv')
                        id = r"2_" + str(k) + r"_" + str(epoch)
                        best_score2 = best_score2[:2]
                        best_score2 += [[ev.quadratic_weighted_kappa(), id]]
                        best_score2.sort(reverse=True)
                        print(best_score1)
                        torch.save(net.state_dict(), r"./save/save_param" + r"/net" + id + r'.ckpt')
                        # pd.DataFrame(threshold.numpy()).to_csv(r"./save/save_param" + r"/threshold" + id + r'.csv', index=None)

                # -- display
                if isbest == True or hp["isSave"] != True:
                    print("==validation===========================")
                    print("epoch: ", epoch)
                    print("acc_total: ", ev.acc_total())
                    print("acc_ave: ", ev.acc_ave())
                    print("kappa: ", ev.quadratic_weighted_kappa())
                    print("acc_0: ", ev.acc_by_cat(cat_num=0))
                    print("acc_1: ", ev.acc_by_cat(cat_num=1))
                    print("acc_2: ", ev.acc_by_cat(cat_num=2))
                    print("acc_3: ", ev.acc_by_cat(cat_num=3))
                    print("diversity: ", ev.diversity())
                    # print("threshold: ", threshold)

                del ev

                # -- test
                if isbest == True or hp["isSave"] != True:
                    out = net(test_ready[0])
                    # ev = calc.EvaluateCat(out, torch.squeeze(test_ready[1]).long())
                    ev = calc.EvaluateCatWithThreshold(out, torch.squeeze(test_ready[1]).long(), threshold)

                    print("==test===========================")
                    print("epoch: ", epoch)
                    print("acc_total: ", ev.acc_total())
                    print("acc_ave: ", ev.acc_ave())
                    print("kappa: ", ev.quadratic_weighted_kappa())
                    print("acc_0: ", ev.acc_by_cat(cat_num=0))
                    print("acc_1: ", ev.acc_by_cat(cat_num=1))
                    print("acc_2: ", ev.acc_by_cat(cat_num=2))
                    print("acc_3: ", ev.acc_by_cat(cat_num=3))
                    print("diversity: ", ev.diversity())
                    # print("threshold: ", threshold)
                    del ev

            isbest = False
            if isdiversity == False:
                print("divesity vanish")
                print("epoch: ", epoch)
                sys.exit()


def main_ev():
    device = "cpu"
    # -- data setting
    reg = regularize.Regularize1(train_data_path_base=r"/train.csv",
                                 test_data_path_base=r"/test.csv")
    test_origin = reg.read_origin_data("test")
    test = reg.regularize(test_origin)
    test_ready = reg.set_input_data(torch.Tensor(test.values), device, "double")

    # -- network setting
    net = ml.Net()
    net.to(device)
    net.double()
    print("device: " + str(device))

    # -- start evaluation
    net.eval()
    with torch.no_grad():
        param_path_list = glob.glob(r"./save/save_param/**.ckpt")
        for i in range(len(param_path_list)):
            param_path = param_path_list[i]
            net.load_state_dict(torch.load(param_path, map_location=device))
            out = net(test_ready[0])
            print(out)
            ev = calc.EvaluateCat(out, torch.squeeze(test_ready[1]).long())
            if i == 0:
                predict_sum = ev.predict
            else:
                predict_sum += ev.predict
            del ev
        predict = predict_sum / len(param_path_list)
        ev = calc.EvaluateCat(predict, torch.squeeze(test_ready[1]).long())
        print("==test===========================")
        print("acc_total: ", ev.acc_total())
        print("acc_ave: ", ev.acc_ave())
        print("kappa: ", ev.quadratic_weighted_kappa())
        print("acc_0: ", ev.acc_by_cat(cat_num=0))
        print("acc_1: ", ev.acc_by_cat(cat_num=1))
        print("acc_2: ", ev.acc_by_cat(cat_num=2))
        print("acc_3: ", ev.acc_by_cat(cat_num=3))
        print("diversity: ", ev.diversity())

    result = pd.concat([test_origin, pd.DataFrame(predict.numpy())], axis=1)
    result.to_csv(r"./save/result/result.csv", encoding="cp932")


if __name__ == "__main__":
    main_ev()
