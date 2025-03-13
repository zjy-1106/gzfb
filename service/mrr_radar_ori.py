#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from matplotlib.dates import DateFormatter, rrulewrapper, RRuleLocator

from module.public.avaliable_product import mrr_image_type_ori
from service.effect_analysis import set_x_time
from service.utils import get_limit_index, transfer_path, ax_add_time_range
# from utils.file_uploader import minio_client

mpl.use("Agg")
mpl.rc("font", family='DengXian')
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
FONTSIZE = 10


class MRROri:
    # 读取原始文件数据
    def __init__(self, start_time, end_time, img_type, height, out_path, before_file, title):
        filepath = before_file[0].get('filePath')
        if start_time and end_time:
            self.start_time = datetime.strptime(start_time[:-2] + '00', '%Y%m%d%H%M%S')
            self.end_time = datetime.strptime(end_time[:-2] + '00', '%Y%m%d%H%M%S')
        else:
            self.start_time = datetime.strptime(os.path.dirname(filepath)[-6:-2]+os.path.basename(filepath)[:4],
                                                '%Y%m%d')
            self.end_time = self.start_time + timedelta(days=1)

        self.img_type = img_type
        self.height = height
        self.title = title
        self.img_path = os.path.join(out_path,
                                     f'{self.img_type}_{self.start_time:%Y%m%d%H%M%S}_{self.end_time:%Y%m%d%H%M%S}.png')
        self.file_time = []  # 监测时间
        self.level = []
        self.raindrop_speed = []  # 雨滴下落速度
        self.raindrop_diameter = []  # 雨滴直径
        # self.raindrop_spectral_distribution_ori = []  # 原始雨滴谱分布
        self.raindrop_spectral_distribution = []  # 计算得到的雨滴谱分布
        self.Z = []  # 订正后的回波强度
        self.RR = []  # 对应高度层雨强
        self.LWC = []  # 含水量
        self.W = []

        for file in before_file:
            with open(transfer_path(file.get('filePath')), 'r', encoding='gbk') as f:
                while 1:
                    data = []
                    i = 0
                    for line in f:
                        data.append(line)
                        i = i + 1
                        if i == 201:
                            break
                    if len(data) < 201:
                        break

                    time_info = data[0].split()[1]
                    curr_time = datetime.strptime(time_info, '%y%m%d%H%M%S')
                    if curr_time < self.start_time:
                        # 只获取指定时间数据
                        continue
                    if curr_time > self.end_time:
                        break
                    self.file_time.append(curr_time)
                    self.level.append([float(item) / 1000 for item in data[1].split()[1:]])
                    tmp_Z_list = []
                    for i in range(0, 217, 7):  # 回波强度
                        tmp_data = data[197][3:][i:i + 7].strip()
                        tmp_data = tmp_data if tmp_data else 0
                        tmp_Z_list.append(tmp_data)
                    self.Z.append(tmp_Z_list)

                    tmp_RR_list = []
                    for i in range(0, 217, 7):  # 雨强
                        tmp_data = data[198][3:][i:i + 7].strip()
                        tmp_data = tmp_data if tmp_data else 0
                        tmp_RR_list.append(tmp_data)
                    self.RR.append(tmp_RR_list)

                    tmp_LWC_data = []
                    for i in range(0, 217, 7):  # 含水量
                        tmp_data = data[199][3:][i:i + 7].strip()
                        tmp_data = tmp_data if tmp_data else 0
                        tmp_LWC_data.append(tmp_data)
                    self.LWC.append(tmp_LWC_data)

                    # 雨滴直径
                    for line in data[67:131]:
                        tmp_diameter_data = []
                        for i in range(0, 217, 7):
                            tmp_data = line[3:][i:i + 7].strip()
                            tmp_data = float(tmp_data) if tmp_data else 0
                            tmp_diameter_data.append(tmp_data)
                        self.raindrop_diameter.append(tmp_diameter_data)

                    # 雨滴谱分布
                    for line in data[131:195]:
                        tmp_spectral_data = []
                        extra_k = 0
                        i = 0
                        while i < 31:
                            # 处理数据不规整问题   could not convert string to float: '0-5.8e+'
                            # for i in range(0, 31):  # 原始数据/1000
                            try:
                                k = i * 7 + extra_k
                                tmp_data = line[3:][k:k + 7]
                                tmp_data = tmp_data.strip()
                                if tmp_data and '-' not in tmp_data[0] and (
                                        len(tmp_data.split('.')[0]) > 1 and 'e' in tmp_data.split('.')[-1]):
                                    raise Exception(f'数据异常{tmp_data}')
                                tmp_data = tmp_data.strip()
                                tmp_data = float(tmp_data) / 1000 if tmp_data else 0
                                tmp_spectral_data.append(tmp_data)
                            except Exception:
                                i = i - 1
                                k = i * 7 + extra_k
                                tmp_spectral_data = tmp_spectral_data[0:-1]
                                tmp_data = line[3:][k:k + 8].strip()
                                tmp_data = float(tmp_data) / 1000 if tmp_data else 0
                                tmp_spectral_data.append(tmp_data)
                                extra_k += 1
                            i += 1
                        self.raindrop_spectral_distribution.append(tmp_spectral_data)

        self.element_map = {
            "echo": self.Z, "raininess": self.RR, "water": self.LWC,
        }
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

    def draw(self, title, ylims=None, eff_time=None, hail_time=None):
        if os.path.isfile(self.img_path):
            return self.img_path
        # 根据img_type分配绘图方法
        if self.img_type == "time_diameter_density":
            return self.draw_with_time(title, eff_time=eff_time, hail_time=hail_time)
        elif self.img_type == "diameter_height_density":
            return self.draw_with_height(title, ylims)
        else:
            return self.draw_with_time_height(title, ylims, eff_time=eff_time, hail_time=hail_time)

    def draw_with_time(self, title, eff_time=None, hail_time=None):
        # 时间-雨滴数大小-雨滴数密度
        _, unit = mrr_image_type_ori.get(self.img_type)
        fig, ax = plt.subplots(figsize=(8, 3))
        title_time = '~'.join([self.file_time[0].strftime("%Y-%m-%d %H:%M"), self.file_time[-1].strftime("%Y-%m-%d %H:%M")])
        ax.set_ylabel("粒径(mm)", fontsize=FONTSIZE)  # xlabel=f"{self.file_time[0]:%Y-%m-%d}",
        ax.set_title(label=f'{title}\t{title_time}', fontsize=FONTSIZE)

        x = []
        y = []
        level = self.height  # 获取高度层索引
        rain_diameter = np.array(self.raindrop_diameter, dtype='float')[:, level].reshape(
            int(len(self.raindrop_diameter) / 64), 64)
        data = np.array(self.raindrop_spectral_distribution, dtype='float')[:, level].reshape(
            int(len(self.raindrop_diameter) / 64), 64)
        data[data <= 0] = None
        for i in range(len(self.file_time)):
            for j in range(rain_diameter.shape[1]):
                x.append(self.file_time[i])
                y.append(rain_diameter[i, j])

        x = np.array(x).reshape(data.shape[0], data.shape[1])
        y = np.array(y).reshape(data.shape[0], data.shape[1])

        # step = get_limit_index(x.shape[0])  # 设置x轴坐标
        # ticks = x[:, 0][::step]
        # plt.xticks(ticks, [f"{item:%H:%M}" for item in ticks])
        kwargs_major, kwargs_minor, timestyle = set_x_time(x[:, 0])
        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)

        bins = [10, 20, 30,40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000, 3000, 5000, 7000, 9000, 10000]
        nbin = len(bins) - 1
        cmap4 = plt.cm.get_cmap('jet', nbin)
        norm4 = mcolors.BoundaryNorm(bins, nbin)

        cset = ax.pcolormesh(x, y, data, cmap=cmap4, norm=norm4, shading='nearest')
        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

        plt.tick_params(labelsize=FONTSIZE)
        cb = plt.colorbar(cset, shrink=0.9, pad=0.05)
        cb.ax.set_title(unit, fontsize=8)
        cb.ax.tick_params(labelsize=8)
        fig.autofmt_xdate(rotation=45)  # 自动调整x轴标签的角度

        plt.savefig(self.img_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(self.img_path)
        return self.img_path

    def draw_with_height(self, title, ylims=None):
        _, unit = mrr_image_type_ori.get(self.img_type)
        # 粒径-高度-雨滴数密度变化图 指定时间
        fig, ax = plt.subplots(figsize=(8, 3))
        title_time = f'{str(self.start_time)}~{str(self.end_time)}'
        ax.set_xlabel("粒径(mm)", fontsize=FONTSIZE)
        ax.set_ylabel("高度(km)", fontsize=FONTSIZE)
        ax.set_title(label=f'{title}\t{title_time}', fontsize=FONTSIZE)

        rain_diameter = np.array(self.raindrop_diameter, dtype='float')
        level = np.array(self.level, dtype=float)
        level = np.tile(level, (64, 1))
        data = np.array(self.raindrop_spectral_distribution, dtype='float')

        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000, 3000, 5000, 7000, 9000, 10000]
        nbin = len(bins) - 1
        cmap4 = plt.cm.get_cmap('jet', nbin)
        norm4 = mcolors.BoundaryNorm(bins, nbin)
        cset = ax.pcolormesh(rain_diameter, level, data, cmap=cmap4, norm=norm4, shading='nearest')
        ax.set_ylim(ylims)

        plt.tick_params(labelsize=FONTSIZE)
        cb = plt.colorbar(cset, shrink=0.9, pad=0.05)
        cb.ax.set_title(unit, fontsize=8)
        cb.ax.tick_params(labelsize=8)

        plt.savefig(self.img_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(self.img_path)
        return self.img_path

    def draw_with_time_height(self, title, ylims=None, eff_time=None, hail_time=None):
        # 回波强度、雨强、含水量，雨滴谱分布参数的时间-高度图
        _, unit = mrr_image_type_ori.get(self.img_type)
        ele = self.img_type.split('_')[-1]
        data = np.array(self.element_map.get(ele), dtype='float')
        if data.shape[0] < 2:
            raise Exception("当前时间筛选数据不足，无法绘制填色图")
        if data.size == 0:
            raise Exception(f"{self.title}文件中没有该时间段数据")

        fig, ax = plt.subplots(figsize=(8, 3))
        title_time = '~'.join([self.file_time[0].strftime("%Y-%m-%d %H:%M"), self.file_time[-1].strftime("%Y-%m-%d %H:%M")])
        ax.set_ylabel("高度(km)", fontsize=FONTSIZE)  #  xlabel=f"{self.file_time[0]:%Y-%m-%d}",
        ax.set_title(label=f'{title}\t{title_time}', fontsize=FONTSIZE)

        x = []
        y = []
        y_value = np.array(self.level)
        for i in range(len(self.file_time)):
            for j in range(y_value.shape[1]):
                x.append(self.file_time[i])
                y.append(y_value[i, j])

        x = np.array(x).reshape(data.shape[0], data.shape[1])
        y = np.array(y).reshape(data.shape[0], data.shape[1])
        # step = get_limit_index(x.shape[0])  # 设置x轴坐标
        # ticks = x[:, 0][::step]
        # plt.xticks(ticks, [f"{item:%H:%M}" for item in ticks])
        kwargs_major, kwargs_minor, timestyle = set_x_time(x[:, 0])
        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)
        # 回波强度

        if ele in ("echo",):
            data[data == -999] = None
            cset = ax.pcolormesh(x, y, data, vmin=-50, vmax=40, cmap="jet", shading='nearest')
            cb = plt.colorbar(cset, shrink=0.9, pad=0.05, format='%.1f')

        elif ele == "density":
            data[data <= 0] = None
            cset = ax.pcolormesh(x, y, data, norm=LogNorm(vmin=1e0, vmax=float(np.nanmax(data))),
                                 cmap="jet", shading='nearest')
            cb = plt.colorbar(cset, shrink=0.9, pad=0.05, format='%.1e')
        elif ele == "raininess":
            data[data <= 0] = None
            if np.nanmax(data) > 100:
                color_level = [0, 1, 5, 10, 25, 50, 100, 250]  # 2, 4, 6
                cmap = plt.cm.get_cmap("jet")
                norm = mcolors.SymLogNorm(1, vmin=color_level[0], vmax=color_level[-1])
                cset = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm, shading='nearest')
                cb = plt.colorbar(cset, shrink=0.9, pad=0.05,
                                  ticks=color_level, format=mpl.ticker.ScalarFormatter(), fraction=0.1)
            else:
                cset = ax.pcolormesh(x, y, data, vmin=np.nanmin(data), vmax=np.nanmax(data), cmap="jet", shading='nearest')
                cb = plt.colorbar(cset, shrink=0.9, pad=0.05, format='%.1f')
        else:
            data[data <= 0] = None
            cset = ax.pcolormesh(x, y, data, vmin=np.nanmin(data), vmax=np.nanmax(data), cmap="jet", shading='nearest')
            cb = plt.colorbar(cset, shrink=0.9, pad=0.05, format='%.1f')

        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

        ax.set_ylim(ylims)
        plt.tick_params(labelsize=FONTSIZE)
        cb.ax.set_title(unit, fontsize=8)
        fig.autofmt_xdate(rotation=45)  # 自动调整x轴标签的角度
        cb.ax.tick_params(labelsize=8)

        plt.savefig(self.img_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(self.img_path)
        return self.img_path
