# -*- coding: utf-8 -*-
import os
import numpy as np
from config.config import ELEMENT_MAP_MRR, font_path
from module.mrr.parse import ParseMRR
import matplotlib as mpl

mpl.use('Agg')


class MRR(ParseMRR):
    def __init__(self):
        super(MRR, self).__init__()
        self.font_dirs = os.path.join(font_path)

    def draw_yq(self, data, h_index):
        """

        :param data: list 数据
        :param h_index: int 高度层
        :return:
        """
        x = []
        y = []
        for d in data:
            x.append(d['01'])
            if x[-1] == '20220727143601':
                pass
            y.append(d["07"])

        x = np.array(x)  # .astype(np.datetime64)

        draw_info = {
            "tabels": ["RR"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'raininess',  # (mm/h)
            "ele": "raininess",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        y = np.array(y)
        y[y == ""] = np.nan
        y = y.astype(float)
        y = np.where(y == -999.0000, 0, y)

        y = y[:, h_index]

        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        new_array = np.array([y])
        return x, new_array, x_label, y_label

    def draw_hsl(self, data, h_index):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["08"])

        x = np.array(x)  # .astype(np.datetime64)
        draw_info = {
            "tabels": ["LWC"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'water',  # (mm/h)
            "ele": "water",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        y = np.array(y)
        y[y == ""] = np.nan
        y = y.astype(float)
        y = np.where(y == -999.0000, 0, y)

        y = y[:, h_index]
        new_array = np.array([y])

        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        return x, new_array, x_label, y_label

    def draw_hb_sj(self, data, h_index):
        """
        32个高度层的经过衰减订正的回波强度，标识符：Z
        :param data:
        :param h_index: 高度层
        :return:
        """
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["06"])

        x = np.array(x)  # .astype(np.datetime64)
        draw_info = {
            "tabels": ["Z"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'echo',  # (mm/h)
            "ele": "echo",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        y = np.array(y)
        y = np.where(y == "", np.nan, y).astype(float)
        y = np.where(y == -999.0000, np.nan, y)

        y = y[:, h_index]
        new_array = np.array([y])
        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        return x, new_array, x_label, y_label

    def draw_hb(self, data, h_index):
        """
        （32个高度层5个波段（Ka，Ku，X，C和S）的回波强度，标识符分别为ZA，ZU，ZX，ZC，ZS
        :param data:
        :param h_index:
        :return:
        """
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["10"])

        x = np.array(x)  # .astype(np.datetime64)
        draw_info = {
            "tabels": ["Ka", "Ku", "X", "C", "S"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'echos',  # (mm/h)
            "ele": "echos",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        y = np.array(y)
        y = np.where(y == "", np.nan, y).astype(float)
        y = np.where(y == -999.0000, np.nan, y)

        ka = y[:, 0][:, h_index]
        ku = y[:, 1][:, h_index]
        k_x = y[:, 2][:, h_index]
        c = y[:, 3][:, h_index]
        s = y[:, 4][:, h_index]

        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        new_array = np.array([ka, ku, k_x, c, s])
        return x, new_array, x_label, y_label

    def draw_ZDR(self, data, h_index):
        draw_info = {
            "tabels": ["Ka", "Ku", "X", "C", "S"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'ZDR',  # (mm/h)
            "ele": "ZDR",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        x = []
        y = []
        for d in data:
            x.append(d['01'])
            y.append(d["11"])

        x = np.array(x)  # .astype(np.datetime64)

        y = np.array(y)
        y[y == ""] = np.nan
        y.astype(float)
        # y = np.where(y == "", np.nan, y).astype(float)
        y[y == -999.0000] = np.nan
        # y = np.where(y == -999.0000, np.nan, y)

        ka = y[:, 0][:, h_index]
        ku = y[:, 1][:, h_index]
        k_x = y[:, 2][:, h_index]
        c = y[:, 3][:, h_index]
        s = y[:, 4][:, h_index]

        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        new_array = np.array([ka, ku, k_x, c, s])
        return x, new_array, x_label, y_label

    def draw_nw(self, data, h_index):
        """
        Gamma分布的数密度
        :param data: list 数据
        :param h_index: int 高度层
        :return:
        """
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["12"])

        x = np.array(x)  # .astype(np.datetime64)
        draw_info = {
            "tabels": ["Nw"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'density',  # (mm/h)
            "ele": "density",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        y = np.array(y)
        y = np.where(y == "", np.nan, y).astype(float)
        y = np.where(y == -999.0000, np.nan, y)

        y = y[:, h_index]
        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        new_array = np.array([y])
        return x, new_array, x_label, y_label

    def draw_dm(self, data, h_index):
        """
        Gamma分布的平均直径Dm
        :param data: list 数据
        :param h_index: int 高度层
        :return:
        """
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["13"])

        x = np.array(x)  # .astype(np.datetime64)
        draw_info = {
            "tabels": ["Dm"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'diameter',  # (mm/h)
            "ele": "diameter",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        y = np.array(y)
        y = np.where(y == "", np.nan, y).astype(float)
        y = np.where(y == -999.0000, np.nan, y)

        y = y[:, h_index]
        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        new_array = np.array([y])
        return x, new_array, x_label, y_label

    def draw_mu(self, data, h_index):
        """
        Gamma分布的形状因子Mu
        :param data: list 数据
        :param h_index: int 高度层
        :return:
        """
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["14"])

        x = np.array(x)  # .astype(np.datetime64)
        draw_info = {
            "tabels": ["Mu"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'Mu',  # (mm/h)
            "ele": "Mu",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        y = np.array(y)
        y = np.where(y == "", np.nan, y).astype(float)
        y = np.where(y == -999.0000, np.nan, y)
        y = y[:, h_index]

        x_label = ''
        y_label = ELEMENT_MAP_MRR[draw_info["ele"]]["unit"]
        new_array = np.array([y])
        return x, new_array, x_label, y_label
