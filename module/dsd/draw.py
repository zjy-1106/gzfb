__author__ = 'panchuang'

import numpy as np
from config.config import font_path, ELEMENT_MAP_DSD
from module.dsd.parse import ParseDSD


class DSD(ParseDSD):
    def __init__(self):
        super(DSD, self).__init__()
        self.dat = []
        self.txt = []
        self.font_dirs = font_path

    def draw_yq(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["07"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, 0, y)

        nw = y[:, 0]

        draw_info = {
            "tabels": ["雨强"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'RAINFALL',  # (mm/h)
            "ele": "raininess",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        new_array = np.array([nw])
        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        return x, new_array, x_label, y_label

    def draw_yq_zkq(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["06"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, 0, y)

        nw = y[:, 0]

        draw_info = {
            "tabels": ["雨强"],
            "colors": [],
            "x_label": 'Time',  # (BT)
            "y_label": 'RAINFALL',  # (mm/h)
            "ele": "rainstrong",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }
        new_array = np.array([nw])

        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        return x, new_array, x_label, y_label

    def draw_hsl(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["07"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, 0, y)

        dm = y[:, 1]

        draw_info = {
            "tabels": ["含水量"],
            "colors": [],
            "x_label": '时间',  # (BT)
            "y_label": 'rlwc',  # (mm/h)
            "ele": "water",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }
        new_array = np.array([dm])
        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        return x, new_array, x_label, y_label

    def draw_hb(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["08"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, np.nan, y)

        ka = y[:, 0]
        ku = y[:, 1]
        k_x = y[:, 2]
        c = y[:, 3]
        s = y[:, 4]

        draw_info = {
            "tabels": ["Ka", "Ku", "X", "C", "S"],
            "colors": [],
            "x_label": '时间',  # (BT)
            "y_label": 'ZH',  # (mm/h)
            "ele": "echo",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }
        new_array = np.array([ka, ku, k_x, c, s])

        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        return x, new_array, x_label, y_label

    def draw_hb_ys(self, data):
        """
        质控前-原始数据回波强度
        :param data:
        :return:
        """
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["06"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = y[:, 1]
        y = np.where(y == -9.999, np.nan, y)

        draw_info = {
            "tabels": ["回波强度"],
            "colors": [],
            "x_label": '时间',  # (BT)
            "y_label": 'ZH',  # (mm/h)
            "ele": "echo",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        new_array = np.array([y])
        return x, new_array, x_label, y_label

    def draw_ZDR(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["09"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, np.nan, y)

        ka = y[:, 0]
        ku = y[:, 1]
        k_x = y[:, 2]
        c = y[:, 3]
        s = y[:, 4]

        draw_info = {
            "tabels": ["Ka", "Ku", "X", "C", "S"],
            "colors": [],
            "x_label": '时间',  # (BT)
            "y_label": 'ZDR',  # (mm/h)
            "ele": "ZDR",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        new_array = np.array([ka, ku, k_x, c, s])
        return x, new_array, x_label, y_label

    def draw_KDP(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["10"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, np.nan, y)

        ka = y[:, 0]
        ku = y[:, 1]
        k_x = y[:, 2]
        c = y[:, 3]
        s = y[:, 4]

        draw_info = {
            "tabels": ["Ka", "Ku", "X", "C", "S"],
            "colors": [],
            "x_label": '时间',  # (BT)
            "y_label": 'KDP',  # (mm/h)
            "ele": "KDP",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        new_array = np.array([ka, ku, k_x, c, s])
        return x, new_array, x_label, y_label

    def draw_sppz(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["11"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, np.nan, y)

        ka = y[:, 0]
        ku = y[:, 1]
        k_x = y[:, 2]
        c = y[:, 3]
        s = y[:, 4]

        draw_info = {
            "tabels": ["Ka", "Ku", "X", "C", "S"],
            "colors": [],
            "x_label": '时间',  # (BT)
            "y_label": 'Lattenuation',  # (mm/h)
            "ele": "Lattenuation",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        new_array = np.array([ka, ku, k_x, c, s])
        return x, new_array, x_label, y_label

    def draw_czpz(self, data):
        x = []
        y = []

        for d in data:
            x.append(d['01'])
            y.append(d["12"])

        x = np.array(x)  # .astype(np.datetime64)
        y = np.array(y, dtype=float)
        y = np.where(y == -999.0000, np.nan, y)

        ka = y[:, 0]
        ku = y[:, 1]
        k_x = y[:, 2]
        c = y[:, 3]
        s = y[:, 4]

        draw_info = {
            "tabels": ["Ka", "Ku", "X", "C", "S"],
            "colors": [],
            "x_label": '时间',  # (BT)
            "y_label": 'Vattenuation',  # (mm/h)
            "ele": "Vattenuation",
            "figsize": (14, 8),
            "xylabel_size": 15,
            "xticks_size": 10,
        }

        x_label = ''
        y_label = ELEMENT_MAP_DSD[draw_info["ele"]]["unit"]
        new_array = np.array([ka, ku, k_x, c, s])
        return x, new_array, x_label, y_label
