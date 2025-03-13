#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    贵州项目解析微雨雷达文件, 质控前数据解析绘图
"""
import os
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import matplotlib.colors as mcolors
from matplotlib.dates import rrulewrapper, RRuleLocator, DateFormatter

from module.public.avaliable_product import mrr_image_type
from service.effect_analysis import set_x_time
from service.utils import get_limit_index, ax_add_time_range
# from utils.file_uploader import minio_client

# 设置中文字体，负号
mpl.rc("font", family='DengXian')
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
mpl.use('Agg')
FONTSIZE = 10


class MRRRadar:
    # 读取文件数据
    def __init__(self, start_time, end_time, img_type, title, out_path, after_file, height=None):
        filepath = after_file[0]  # 测试文件
        if start_time and end_time:
            self.start_time = datetime.strptime(start_time[:-2], '%Y%m%d%H%M')
            self.end_time = datetime.strptime(end_time[:-2], '%Y%m%d%H%M')
        else:
            # 默认返回文件最后两小时的数据
            file_time = os.path.basename(filepath)[:6]
            self.start_time = datetime.strptime(file_time, '%y%m%d')
            self.end_time = self.start_time + timedelta(days=1)

        self.img_type = img_type
        self.title = title
        self.img_path = os.path.join(out_path,
                                     f'{self.img_type}_{self.start_time:%Y%m%d%H%M%S}_{self.end_time:%Y%m%d%H%M%S}.png')
        self.height = height
        self.file_time = []  # 监测时间
        self.level = []
        self.raindrop_diameter = []  # 雨滴直径
        # self.raindrop_spectral_distribution_ori = []  # 原始雨滴谱分布
        self.raindrop_spectral_distribution = []  # 计算得到的雨滴谱分布
        self.reflectivity = []  # 订正后的回波强度
        self.reflectivity_ori = []  # 回波强度
        self.RR = []  # 对应高度层雨强
        self.LWC = []  # 含水量
        self.W = []  # 雨滴下落速度
        self.ZA = []
        self.ZU = []
        self.ZX = []
        self.ZC = []
        self.ZS = []
        self.DRA = []
        self.DRU = []
        self.DRX = []
        self.DRC = []
        self.DRS = []
        self.Nw = []
        self.Dm = []
        self.Mu = []

        for file in after_file:
            with open(file, 'r', encoding='gbk') as f:
                while 1:
                    data = []
                    i = 0
                    for line in f:
                        data.append(line.strip())
                        i = i + 1
                        if i == 148:
                            break
                    if len(data) < 148:
                        break

                    time_info = data[0].split()
                    curr_time = datetime(year=int('20' + time_info[1]), month=int(time_info[2]), day=int(time_info[3]),
                                         hour=int(time_info[4]), minute=int(time_info[5]))
                    if curr_time < self.start_time:
                        # 只获取指定时间数据
                        continue
                    if curr_time > self.end_time:
                        break
                    self.file_time.append(curr_time)
                    self.level.append([float(item) / 1000 for item in data[1].split()[1:]])
                    for raindrop_diameter in data[2:66]:  # 雨滴直径
                        self.raindrop_diameter.append([float(item) for item in raindrop_diameter.split()[1:32]])
                    for raindrop_density in data[66:130]:  # 雨滴数密度
                        self.raindrop_spectral_distribution.append(
                            [float(item) for item in raindrop_density.split()[1:32]])

                    self.reflectivity_ori.append(
                        [float(item.replace('-999.0', '0.000')) for item in data[130].split()[1:32]])
                    self.reflectivity.append(
                        [float(item.replace('-999.0', '0.000')) for item in data[131].split()[1:32]])
                    self.RR.append([float(item) if "*" not in item else np.nan for item in data[132].split()[1:32]])
                    self.LWC.append([float(item) if "*" not in item else np.nan for item in data[133].split()[1:32]])
                    self.W.append([float(item) if "*" not in item else np.nan for item in data[134].split()[1:32]])
                    self.ZA.append([float(item) if "*" not in item else np.nan for item in data[135].split()[1:32]])
                    self.ZU.append([float(item) if "*" not in item else np.nan for item in data[136].split()[1:32]])
                    self.ZX.append([float(item) if "*" not in item else np.nan for item in data[137].split()[1:32]])
                    self.ZC.append([float(item) if "*" not in item else np.nan for item in data[138].split()[1:32]])
                    self.ZS.append([float(item) if "*" not in item else np.nan for item in data[139].split()[1:32]])
                    self.DRA.append([float(item) if "*" not in item else np.nan for item in data[140].split()[1:32]])
                    self.DRU.append([float(item) if "*" not in item else np.nan for item in data[141].split()[1:32]])
                    self.DRX.append([float(item) if "*" not in item else np.nan for item in data[142].split()[1:32]])
                    self.DRC.append([float(item) if "*" not in item else np.nan for item in data[143].split()[1:32]])
                    self.DRS.append([float(item) if "*" not in item else np.nan for item in data[144].split()[1:32]])
                    self.Nw.append([float(item) if "*" not in item else np.nan for item in data[145].split()[1:32]])
                    self.Dm.append([float(item) if "*" not in item else np.nan for item in data[146].split()[1:32]])
                    self.Mu.append([float(item) if "*" not in item else np.nan for item in data[147].split()[1:32]])

        if not self.file_time:
            raise Exception(f"{self.title}没有该时间段的数据{self.start_time:%Y%m%d%H%M}-{self.end_time:%Y%m%d%H%M}")
        self.element_map = {
            "echo": self.reflectivity, "Ka": self.ZA, "Ku": self.ZU, "X": self.ZX, "C": self.ZC, "S": self.ZS,
            "raininess": self.RR, "water": self.LWC, "density": self.Nw, "diameter": self.Dm,
            "Mu": self.Mu, "W": self.W
        }
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

    def draw(self, title, ylims=None, eff_time=None, hail_time=None):
        # 根据img_type分配绘图方法
        if self.img_type == "time_diameter_density":
            return self.draw_with_time(title, eff_time=eff_time, hail_time=hail_time)
        elif self.img_type == "diameter_height_density":
            return self.draw_with_height(title, ylims)
        else:
            return self.draw_with_time_height(title, ylims, eff_time=eff_time, hail_time=hail_time)

    def draw_with_time(self, title, eff_time=None, hail_time=None):
        # 时间-雨滴数大小-雨滴数密度
        _, unit = mrr_image_type.get(self.img_type)
        fig, ax = plt.subplots(figsize=(8, 3))
        title_time = '~'.join([self.file_time[0].strftime("%Y-%m-%d %H:%M"), self.file_time[-1].strftime("%Y-%m-%d %H:%M")])
        ax.set_ylabel("粒径(mm)", fontsize=FONTSIZE)  # xlabel=f"{self.file_time[0]:%Y-%m-%d}"
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

        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000, 3000, 5000, 7000, 9000, 10000]
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
        _, unit = mrr_image_type.get(self.img_type)
        # 粒径-高度-雨滴数密度变化图 指定时间
        fig, ax = plt.subplots(figsize=(8, 3))
        title = f'{title}\t{str(self.start_time)}~{str(self.end_time)}'
        ax.set_xlabel("粒径(mm)", fontsize=FONTSIZE)
        ax.set_ylabel("高度(km)", fontsize=FONTSIZE)
        ax.set_title(label=title, fontsize=FONTSIZE)

        rain_diameter = np.array(self.raindrop_diameter, dtype='float')
        level = np.array(self.level, dtype=float)
        level = np.tile(level, (64, 1))
        data = np.array(self.raindrop_spectral_distribution, dtype='float')

        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000, 3000, 5000, 7000, 9000, 10000]
        nbin = len(bins) - 1
        cmap4 = plt.cm.get_cmap('jet', nbin)
        norm4 = mcolors.BoundaryNorm(bins, nbin)

        data[data <= 0] = np.nan
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
        _, unit = mrr_image_type.get(self.img_type)
        ele = self.img_type.split('_')[-1]
        data = np.array(self.element_map.get(ele), dtype='float')
        if data.shape[0] < 2:
            raise Exception("当前时间筛选数据不足，无法绘制填色图")
        y_value = np.array(self.level)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_ylabel("高度(km)", fontsize=FONTSIZE)
        if title:
            title_time = '~'.join([self.file_time[0].strftime("%Y-%m-%d %H:%M"), self.file_time[-1].strftime("%Y-%m-%d %H:%M")])
            ax.set_title(f'{title}\t{title_time}', fontsize=FONTSIZE)
        # ax.set_xlabel(f"{self.file_time[0]:%Y-%m-%d}")

        x = []
        y = []
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
        if ele in ("echo", "S", "Ka", "Ku", "X", "C"):
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
        cb.ax.set_title(unit, fontsize=8)
        fig.autofmt_xdate(rotation=45)  # 自动调整x轴标签的角度
        plt.tick_params(labelsize=FONTSIZE)
        cb.ax.tick_params(labelsize=8)

        plt.savefig(self.img_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(self.img_path)
        return self.img_path


if __name__ == '__main__':
    dsd = MRRRadar("", "", "diameter_height_density", "test",
                   "D:/download/220717mrr.dat", 20)
    dsd.draw("雨滴直径-下落速度-雨滴分布")
