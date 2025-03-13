#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import re
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter, rrulewrapper, RRuleLocator, MinuteLocator

from module.public.avaliable_product import dsd_image_type
from service.effect_analysis import set_x_time
from service.utils import get_limit_index, merge_data, ax_add_time_range
# from utils.file_uploader import minio_client

# 设置中文字体，负号
mpl.rc("font", family='DengXian')
mpl.rcParams['axes.unicode_minus'] = False
mpl.use('Agg')
FONTSIZE = 10


class DsdRadar:
    def __init__(self, start_time, end_time, img_type, after_file, title):
        self.filepath = after_file  # 测试文件
        if start_time and end_time:
            self.start_time = datetime.strptime(start_time[:-2] + '00', '%Y%m%d%H%M%S')
            self.end_time = datetime.strptime(end_time[:-2] + '00', '%Y%m%d%H%M%S')
        else:
            file_time = os.path.basename(self.filepath[0])[:8]
            self.start_time = datetime.strptime(file_time, '%Y%m%d')
            self.end_time = self.start_time + timedelta(minutes=1439)
        self.img_type = img_type
        self.title = title
        self.file_time_ori = []  # 监测时间
        self.file_time = []   #
        self.raindrop_speed = []  # 雨滴下落速度
        self.raindrop_diameter = []  # 雨滴直径
        self.raindrop_spectral_distribution_ori = []  # 原始雨滴谱分布
        self.raindrop_spectral_distribution = []  # 计算得到的雨滴谱分布
        self.intensity_and_reflectivity = []  # 雨强, 回波强度
        self.cal_intensity = []  # 计算得到的雨强
        self.ZH = []  # Ka, Ku, X, C, S 波段回波强度ZH
        self.ZDR = []  # Ka, Ku, X, C, S 波段ZDR
        self.KDP = []  # Ka, Ku, X, C, S 波段KDP
        self.HPAO = []  # 水平偏振衰减系数
        self.VPAO = []  # 垂直偏深衰减系数
        self.grade_distribution = []
        self.is_loadfile = False

    def load_file(self):
        if self.is_loadfile:
            return
        for file in self.filepath:
            with open(file, 'rb') as f:
                tag = False
                for line in f:
                    data = re.split(r"\s+", line.decode('utf-8').strip())
                    if data[0] == '01:':
                        curr_time = datetime(year=int(data[1]), month=int(data[2]), day=int(data[3]),
                                             hour=int(data[4]), minute=int(data[5]))
                        if curr_time < self.start_time:
                            continue
                        if curr_time > self.end_time:
                            break
                        self.file_time_ori.append(curr_time)
                        tag = True
                    if tag:
                        if data[0] == '02:':
                            self.raindrop_speed.append(data[1:])
                        elif data[0] == '03:':
                            if not self.raindrop_diameter:
                                self.raindrop_diameter = data[1:]
                        elif data[0] == '04:':
                            self.raindrop_spectral_distribution_ori.append(data[1:])
                        elif data[0] == '05:':
                            self.raindrop_spectral_distribution.append(data[1:])
                        elif data[0] == '06:':
                            self.intensity_and_reflectivity.append(data[1:])
                        elif data[0] == '07:':
                            self.cal_intensity.append(data[1:])
                        elif data[0] == '08:':
                            self.ZH.append(data[1:])
                        elif data[0] == '09:':
                            self.ZDR.append(data[1:])
                        elif data[0] == '10:':
                            self.KDP.append(data[1:])
                        elif data[0] == '11:':
                            self.HPAO.append(data[1:])
                        elif data[0] == '12:':
                            self.VPAO.append(data[1:])
                        elif data[0] == '13:':
                            self.grade_distribution.append(data[1:])
        self.is_loadfile = True
        if not len(self.file_time_ori):
            raise Exception(f"{self.title}没有该时间段数据")

        # 数据按分钟级补全
        time_range1 = pd.date_range(self.start_time, self.end_time, freq='min')
        distribution_df = merge_data(time_range1, self.file_time_ori, self.raindrop_spectral_distribution)
        self.raindrop_spectral_distribution = np.array(distribution_df.iloc[:, 1:], dtype=float)
        distribution_ori_df = merge_data(time_range1, self.file_time_ori, self.raindrop_spectral_distribution_ori)
        self.raindrop_spectral_distribution_ori = np.array(distribution_ori_df.iloc[:, 1:], dtype=float)
        rain_speed_df = merge_data(time_range1, self.file_time_ori, self.raindrop_speed)
        raindrop_speed = np.array(rain_speed_df.iloc[:, 1:], dtype=float)
        self.raindrop_speed = np.nan_to_num(raindrop_speed)

        grade_distribution_df = merge_data(time_range1, self.file_time_ori, self.grade_distribution)
        self.grade_distribution = np.array(grade_distribution_df.iloc[:, 1:], dtype=float)
        self.file_time = distribution_df.data_time.tolist()

    def draw(self, out_path, title, qc=False, density_time=None, eff_time=None, hail_time=None):
        start_t = self.start_time.strftime("%Y%m%d%H%M%S")
        end_t = self.end_time.strftime("%Y%m%d%H%M%S")
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        if qc:
            img_path = os.path.join(out_path, f'{self.img_type}_{start_t}_{end_t}_qc.png')
        else:
            img_path = os.path.join(out_path, f'{self.img_type}_{start_t}_{end_t}_raw.png')

        self.load_file()
        if qc:
            data = np.array(self.raindrop_spectral_distribution, dtype=float)
        else:
            data = np.array(self.raindrop_spectral_distribution_ori, dtype=float)
        data[data <= 0] = np.nan
        rain_diameter = np.array(self.raindrop_diameter, dtype='float')

        _, unit = dsd_image_type.get(self.img_type)
        fig, ax = plt.subplots(figsize=(8, 3))
        if title:
            title_time = '~'.join([self.file_time[0].strftime("%Y-%m-%d %H:%M"), self.file_time[-1].strftime("%Y-%m-%d %H:%M")])
            ax.set_title(f'{title}\t{title_time}', fontsize=FONTSIZE)

        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000, 3000, 5000, 7000, 9000, 10000]
        nbin = len(bins) - 1
        cmap4 = plt.cm.get_cmap('jet', nbin)
        norm4 = mcolors.BoundaryNorm(bins, nbin)

        if self.img_type == 'time_rainspeed_density':
            ax.set_ylabel("下落速度(m/s)", fontsize=FONTSIZE)
            # ax.set_xlabel(f"{self.file_time[0]:%Y-%m-%d}")
            x = np.tile(np.array(self.file_time, dtype=datetime).reshape(len(self.file_time), 1),
                        (1, data.shape[1]))
            y = np.array(self.raindrop_speed, dtype=float)
            # step = get_limit_index(x.shape[0])  # 设置x轴坐标
            # ticks = x[:, 0][::step]
            # plt.xticks(ticks, [str(item)[11:16] for item in ticks])
            kwargs_major, kwargs_minor, timestyle = set_x_time(x)
            majorformatter = DateFormatter(timestyle)
            rule1 = rrulewrapper(**kwargs_major)
            loc1 = RRuleLocator(rule1)
            ax.xaxis.set_major_locator(loc1)
            ax.xaxis.set_major_formatter(majorformatter)
            rule = rrulewrapper(**kwargs_minor)
            loc = RRuleLocator(rule)
            ax.xaxis.set_minor_locator(loc)

            cset = ax.pcolormesh(x, y, data, cmap=cmap4, norm=norm4, shading='nearest')
            fig.autofmt_xdate(rotation=45)
        elif self.img_type == 'time_raindrops_density':
            idx = np.max(np.argwhere(self.raindrop_speed > 0), axis=0)[1]
            ax.set_ylabel("粒径(mm)", fontsize=FONTSIZE)
            # ax.set_xlabel(f"{self.file_time[0]:%Y-%m-%d}")
            ax.set_ylim([0, int(rain_diameter[idx]) + 1])
            x = np.array(self.file_time, dtype=datetime)
            y = rain_diameter
            # x[0].astype(datetime)
            # step = get_limit_index(x.shape[0])  # 设置x轴坐标
            # ticks = x[::step]
            # plt.xticks(ticks, [str(item)[11:16] for item in ticks])

            kwargs_major, kwargs_minor, timestyle = set_x_time(x)
            majorformatter = DateFormatter(timestyle)
            rule1 = rrulewrapper(**kwargs_major)
            loc1 = RRuleLocator(rule1)
            ax.xaxis.set_major_locator(loc1)
            ax.xaxis.set_major_formatter(majorformatter)
            rule = rrulewrapper(**kwargs_minor)
            loc = RRuleLocator(rule)
            ax.xaxis.set_minor_locator(loc)

            # cset = ax.contourf(x, y, data.T, cmap=cmap4, norm=norm4)
            cset = ax.pcolormesh(x, y, data.T, cmap=cmap4, norm=norm4, shading='nearest')
            ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
            ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

            fig.autofmt_xdate(rotation=45)
        elif self.img_type == 'diameter_speed_density':   # 雨滴直径-下落速度图
            # 从有数据的时间中提取完整数据索引
            if density_time:
                pic_time = datetime.strptime(density_time, "%Y%m%d%H%M%S")
            else:
                pic_time = self.file_time_ori[0]
            idx = self.file_time.index(pic_time)
            data = np.array(self.grade_distribution, dtype=float)
            data = data[idx].reshape(32, 32)
            data[data <= 0] = np.nan
            x = rain_diameter
            y = np.array(self.raindrop_speed, dtype=float)[idx]
            # ax.set(xlabel="粒径(mm)", ylabel="下落速度(m/s)", fontsize=6)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 12)
            # ax.set_ylim(0, int(np.max(y)) + 1)
            ax.set_xlabel(xlabel="粒径(mm)", fontsize=FONTSIZE)
            ax.set_ylabel("下落速度(m/s)", fontsize=FONTSIZE)
            # ax.set_title(f'{title}\n{pic_time}', fontsize=6)
            cset = ax.pcolormesh(x, y, data, cmap="jet", shading='auto')
            x_c = np.linspace(0, 10)
            y_c = [9.65 - 10.30 * math.exp(-0.6 * i) for i in x_c]
            ax.plot(x_c, y_c, c='k')
            fig.autofmt_xdate(rotation=0)

            title_time = self.file_time[idx].strftime("%Y-%m-%d %H:%M")
            ax.set_title(f'{title}\t{title_time}', fontsize=FONTSIZE)
            # 就是求平均，每一个直径上有许多和不同速度的滴谱数密度(颜色表示的值)，将这些值和对应的速度相乘，不再相加，然后除以总数密度
            # data1 = np.nan_to_num(data)
            # y.resize(len(y), 1)
            # data_dot = data1.dot(y)
            # data_dot.resize(1, len(y))
            # y1 = data_dot / np.nansum(data1, axis=0)

            # plt.plot(x, y1)

        # 自动调整x轴标签的角度
        plt.tick_params(labelsize=FONTSIZE)
        cb = plt.colorbar(cset, shrink=0.9, pad=0.04)
        cb.ax.set_title(unit, fontsize=8, position=(0.7, 0))
        cb.ax.tick_params(labelsize=8)

        plt.savefig(img_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # # minio_client.put_object(img_path)
        return img_path, self.file_time_ori


if __name__ == '__main__':
    dsd = DsdRadar("", "", "diameter_speed_density",
                   "D:/Documents/dsd/20220805000000DSDPara.dat")
    dsd.draw('test', "雨滴直径-下落速度-雨滴分布", qc=True)
