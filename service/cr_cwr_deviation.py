# -*- coding: utf-8 -*-
import os
from typing import Any

import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

from service.cr import CRController, CR
from service.cwr import CWRController, CWRData
from service.dsd import DsdController
from service.dsd_radar import DsdRadar
from service.mrr import MrrController
from service.mrr_radar import MRRRadar


def calculate_rmse(test, predict):
    try:
        RMSE_cwr_dsd = sqrt(mean_squared_error(test, predict))
    except ValueError:
        RMSE_cwr_dsd = 0
    return RMSE_cwr_dsd


def serch_index_h(stand_h: list, select_h: list):
    """

    :param stand_h: 选择的高度
    :param select_h: 被挑选的高度
    :return:
    """
    ind_h_stand = []
    ind_h_select = []
    for i in stand_h:
        try:
            ind_h_select.append(select_h.index(i))
            ind_h_stand.append(stand_h.index(i))
        except ValueError:
            continue
    return ind_h_stand, ind_h_select


def merge_data(time1, time2, data1, data2):
    time1 = pd.to_datetime(time1)
    time2 = pd.to_datetime(time2)
    new_idx = []
    new_time2 = []
    for i in range(len(time1)):
        try:
            idx = time2.indexer_between_time(time1[i].time(), time1[i + 1].time(), include_start=True,
                                             include_end=False)
        except ValueError:
            continue
        except IndexError:
            idx = time2.indexer_between_time(time1[i].time(), (time1[i] + timedelta(minutes=1)).time(),
                                             include_start=True, include_end=False)
        if idx.any():
            new_idx.append(idx[0])
            new_time2.append(time1[i])

    if len(data1.shape) == 1:
        data_list = ['height0']
        adc1 = np.array([time1] + [data1.T.tolist()], dtype=object).T
        if new_idx:
            data2 = data2[new_idx]
            time2 = new_time2
        adc2 = np.array([time2] + [data2.T.tolist()], dtype=object).T
        len_h = 1
    else:
        data_list = [f'height{i}' for i in range(len(data1[:, 0]))]
        if new_idx:
            data2 = data2[:, new_idx]
            time2 = new_time2
        adc1 = np.array([time1] + [i.T.tolist() for i in data1], dtype=object).T
        adc2 = np.array([time2] + [i.T.tolist() for i in data2], dtype=object).T
        len_h = data1.shape[0]
    df1 = pd.DataFrame(columns=['data_time'] + data_list, data=adc1)
    df2 = pd.DataFrame(columns=['data_time'] + data_list, data=adc2)

    new_df = pd.merge(left=df1, right=df2, how='left', on='data_time')
    res1 = np.array([new_df[f'height{i}_x'] for i in range(len_h)])
    res2 = np.array([new_df[f'height{i}_y'] for i in range(len_h)])

    return new_df, res1, res2


class Deviation(object):
    """雨滴谱仪、微雨雷达、云雷达、C波段连续波雷达误差分析"""

    def __init__(self, fileinfo, start_time, end_time):
        self.fileinfo = fileinfo
        self.start_time = start_time  # '20230810220000'
        self.end_time = end_time  # '20230810230000'
        self.dsd_data: Any = None
        self.dsd_data_zh: Any = None
        self.cr_data: Any = None
        self.cwr_data: Any = None
        self.mrr_data: Any = None
        self.load_run_exe()

    def load_run_exe(self):
        # 雨滴谱仪
        dsd_input_filelist = self.fileinfo['ydpFiles']
        dsd = DsdController(dsd_input_filelist, 1, 'ref', self.start_time, self.end_time)
        dsd.run_exe()
        self.dsd_data = DsdRadar(self.start_time, self.end_time, dsd.element, dsd.data_file_list, '误差分析')

        # 云雷达
        cr_input_filelist = [self.fileinfo['crThiQcFiles']]
        cr = CRController(cr_input_filelist, 2, 'ref', '误差分析', 4)
        cr.exe()
        for f in cr.prod_files:
            f_type = os.path.basename(f).split(".")[0][-4:]  # 数据类型（prod：极坐标数据，Grid：网格数据）
            cr_load = CR(f, f_type, cr.title)
            self.cr_data = cr_load.get_data('refqc')

        # C波段
        cwr_input_filelist = self.fileinfo['cwgFiles']
        cwr = CWRController(cwr_input_filelist, 300, 3, 'ref', '误差分析', 1)
        cwr.run_exe()
        self.cwr_data = CWRData(cwr.output_files, self.start_time, self.end_time)

        # 微雨雷达
        mrr_input_filelist = self.fileinfo['mmrFiles']
        mrr = MrrController(mrr_input_filelist, 300, 4, 'ref', self.start_time, self.end_time)
        mrr.run_exe()
        self.mrr_data = MRRRadar(mrr.start_time, mrr.end_time, mrr.element, '误差分析',
                                 os.path.join(mrr.pic_file_path, "new"), mrr.data_file_list, mrr.height_index)

    def calculate(self):
        _ = np.array(eval("self.cwr_data.{}".format('qc_ref')), dtype=float)
        cwr_h = list(range(self.cwr_data.lib_length[0], _.shape[1] * self.cwr_data.lib_length[0] + 1,
                           self.cwr_data.lib_length[0]))
        cwr_h = [y_km / 1000 for y_km in cwr_h]

        mrr_h_list, cr_h_list = serch_index_h(self.mrr_data.level[0], self.cr_data[1].tolist())
        cr_data_s = self.cr_data[2][cr_h_list, :]
        cr_data_s[cr_data_s == -999] = None
        mrr_data_s = np.array(self.mrr_data.ZA)[:, mrr_h_list].T
        mrr_data_s[mrr_data_s == -999] = None
        red, y_test, y_predict = merge_data(self.mrr_data.file_time, self.cr_data[0], mrr_data_s, cr_data_s)
        RMSE_cr_mrr = calculate_rmse(y_test, y_predict)

        mrr_h_list, cwr_h_list = serch_index_h(self.mrr_data.level[0], cwr_h)
        cwr_data_s = np.array(self.cwr_data.qc_ref, dtype=float)[:, cwr_h_list].T
        cwr_data_s[cwr_data_s == -999] = None
        mrr_data_s = np.array(self.mrr_data.ZA)[:, mrr_h_list].T
        mrr_data_s[mrr_data_s == -999] = None
        red1, y_test1, y_predict1 = merge_data(self.mrr_data.file_time, self.cwr_data.time_list, mrr_data_s, cwr_data_s)
        RMSE_cwr_mrr = calculate_rmse(y_test1, y_predict1)

        if self.dsd_data.file_time_ori:
            try:
                self.dsd_data.load_file()
                dsd_data_zh = np.array(self.dsd_data.ZH, dtype=float)  # 分别对应ka ku X C S 波段
            except Exception:
                dsd_data_zh = None

            cr_data_s = self.cr_data[2][self.cr_data[1].tolist().index(0.3), :]
            cr_data_s[cr_data_s == -999] = None
            red2, y_test2, y_predict2 = merge_data(self.dsd_data.file_time_ori, self.cr_data[0], dsd_data_zh[:, 0],
                                                   cr_data_s)
            RMSE_cr_dsd = calculate_rmse(y_test2, y_predict2)

            cwr_data_s = np.array(self.cwr_data.qc_ref, dtype=float)[:, cwr_h.index(0.3)]
            cwr_data_s[cwr_data_s == -999] = None
            red3, y_test3, y_predict3 = merge_data(self.dsd_data.file_time_ori, self.cwr_data.time_list, dsd_data_zh[:, 3],
                                                   cwr_data_s)
            RMSE_cwr_dsd = calculate_rmse(y_test3, y_predict3)
        else:
            RMSE_cr_dsd, RMSE_cwr_dsd = 0, 0

        return np.nanmean([RMSE_cr_mrr, RMSE_cr_dsd]), np.nanmean([RMSE_cwr_mrr, RMSE_cwr_dsd])

# Deviation('fff', '20230810220000', '20230810230000')
