import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Union, Dict
from abc import abstractmethod


class Regularize():

    @abstractmethod
    def read_origin_data(self, test_or_train_flag: str = "train") -> pd.DataFrame:
        pass

    @abstractmethod
    def regularize(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def set_input_data(data: torch.Tensor, device: str = "cpu", dtype: str = "double") -> List[torch.Tensor]:
        """
        modelにinputするためにdataをcagtegory毎に分け，型を統一し，使用するdeviceを設定する \n
        """
        pass

    @staticmethod
    def split_valid_train(data: torch.Tensor, cross_num: int, cross_id: int, seed: int) -> List[torch.Tensor]:
        sample_num = len(data)
        indexes = np.arange(sample_num)
        np.random.seed(seed)
        np.random.shuffle(indexes)

        batch_size = int(sample_num / cross_num)
        if batch_size * cross_id <= sample_num:
            val_indexes = indexes[batch_size * cross_id:batch_size * (cross_id + 1)].tolist()
        else:
            val_indexes = indexes[batch_size * cross_id:].tolist()
        train_indexes = list(set(indexes) - set(val_indexes))

        train_data = data[train_indexes]
        val_data = data[val_indexes]

        return [train_data, val_data]

    @staticmethod
    def split_in_bool(data: pd.DataFrame, bool_col_num: int) -> List[pd.DataFrame]:
        """
        融資可否フラグに沿ってデータサンプルをsplitする \n
        :
        return: data0 which is label_bool=0, data1 which is label_bool=1
        """
        data0_list = []
        data1_list = []
        for i in range(data.shape[0]):
            if data.iat[i, bool_col_num] == 0:
                data0_list += [data.iloc[i, :].values.tolist()]
            elif data.iat[i, bool_col_num] == 1:
                data1_list += [data.iloc[i, :].values.tolist()]
            else:
                print("ERROR in split_in_bool")
                sys.exit()

        data0 = pd.DataFrame(data0_list)
        data1 = pd.DataFrame(data1_list)
        return [data0, data1]


class Regularize1(Regularize):
    def __init__(self, train_data_path_base, test_data_path_base):
        self.train_data_path_base = train_data_path_base
        self.test_data_path_base = test_data_path_base

    def read_origin_data(self, test_or_train_flag: str = "train") -> pd.DataFrame:
        """
        :param test_or_train_flag: "train" or "test" \n
        :return: data for train or test \n
        """
        if test_or_train_flag == "train":
            return pd.read_csv(r"./data" + self.train_data_path_base, encoding="cp932")
        elif test_or_train_flag == "test":
            return pd.read_csv(r"./data" + self.test_data_path_base, encoding="cp932")
        else:
            print("ERROR in regularize.Regularize1.read_origin_data")
            sys.exit()

    def regularize(self, data: pd.DataFrame) -> pd.DataFrame:
        mu_list = [8.328833333,
                   0.5229625,
                   0.275,
                   2.529708333,
                   0.087471667,
                   15.8325,
                   46.77791667,
                   0.996750458,
                   3.309225,
                   0.659308333,
                   10.41034722]

        std_list = [1.718205119,
                    0.177321239,
                    0.194399913,
                    1.402004224,
                    0.048147769,
                    10.47095082,
                    33.67286156,
                    0.001878597,
                    0.152292406,
                    0.173877733,
                    1.069432151]

        for i in range(11):
            data.iloc[:, i] = (data.iloc[:, i] - mu_list[i]) / std_list[i]

        for i in range(data.shape[0]):
            if data.iat[i, 11] == 3.0:
                data.iat[i, 11] = 0
            elif data.iat[i, 11] == 4.0:
                data.iat[i, 11] = 0
            elif data.iat[i, 11] == 5.0:
                data.iat[i, 11] = 1
            elif data.iat[i, 11] == 6.0:
                data.iat[i, 11] = 2
            elif data.iat[i, 11] == 7.0:
                data.iat[i, 11] = 3
            else:
                data.iat[i, 11] = 3

        return data

    @staticmethod
    def set_input_data(data: torch.Tensor, device: str = "cpu", dtype: str = "double") -> List[torch.Tensor]:
        """
        modelにinputするためにdataをcagtegory毎に分け，型を統一し，使用するdeviceを設定する \n
        """
        data, label = torch.split(data, [11, 1], dim=1)
        if dtype == "double":
            data = data.double().to(device)
            label = torch.squeeze(label).long().to(device)
        else:
            print("ERROR in regularize.Regularize1.set_input_data")
            print("this dtype not implemented yet")
            sys.exit()

        return [data, label]

    @staticmethod
    def split_in_cat(data: pd.DataFrame, cat_col_num: int) -> List[pd.DataFrame]:
        """
        カテゴリーに沿ってデータサンプルをsplitする \n
        :
        return: data_i which is label_cal = i
        """
        data0_list = []
        data1_list = []
        data2_list = []
        data3_list = []
        for i in range(data.shape[0]):
            if data.iat[i, cat_col_num] == 0:
                data0_list += [data.iloc[i, :].values.tolist()]
            elif data.iat[i, cat_col_num] == 1:
                data1_list += [data.iloc[i, :].values.tolist()]
            elif data.iat[i, cat_col_num] == 2:
                data2_list += [data.iloc[i, :].values.tolist()]
            elif data.iat[i, cat_col_num] == 3:
                data3_list += [data.iloc[i, :].values.tolist()]
            else:
                print("ERROR in split_in_bool")
                sys.exit()

        data0 = pd.DataFrame(data0_list)
        data1 = pd.DataFrame(data1_list)
        data2 = pd.DataFrame(data2_list)
        data3 = pd.DataFrame(data3_list)
        return [data0, data1, data2, data3]
