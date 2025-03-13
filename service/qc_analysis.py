#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取质控报告相关绘图
"""

import os
import glob
from datetime import datetime
from pathlib import Path
from math import sin, cos, atan2, pi

import geopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.colors import ListedColormap, BoundaryNorm
from netCDF4 import Dataset
from geopy.distance import geodesic

from config.config import config_info
from service.colormap import colormap
from service.effect_analysis import Effect_Analysis
from service.utils import transfer_path, get_lon_lat
# from utils.file_uploader import minio_client

mpl.rc("font", family='Microsoft YaHei')
mpl.rcParams['axes.unicode_minus'] = False

eles_pol = ['ref', 'vel', 'wid', 'zdr', 'rhv', 'pdp', 'kdp', 'snr', 'scw']  # , 'hcl', 'ccl'
eles_grb = ['ref', 'zdr', 'rhv', 'kdp', 'hcl', 'ccl', 'scw']


def load_data(data1):
    with open("D:/b.txt", 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write('[')
        for indx in range(0, 1500):
            f.write('[')
            for indy in range(0, 64):
                if indy == 63:
                    f.write(f'{data1[indx, indy].astype(int)}')
                else:
                    f.write(f'{data1[indx, indy].astype(int)},')
            f.write('],\n')
        f.write(']')


def calcu_azimuth(lat1, lon1, lat2, lon2):
    """
    计算两经纬度连线的方位
    :param lat1: 起始纬度点
    :param lon1: 起始经度点
    :param lat2: 结束纬度点
    :param lon2: 结束经度点
    :return:
    """
    lat1_rad = lat1 * pi / 180
    lon1_rad = lon1 * pi / 180
    lat2_rad = lat2 * pi / 180
    lon2_rad = lon2 * pi / 180
    y = sin(lon2_rad - lon1_rad) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(lon2_rad - lon1_rad)
    brng = atan2(y, x) / pi * 180
    return float((brng + 360.0) % 360.0)


def extend_point(start_lat, start_lon, end_lat, end_lon, distance=1):
    """
    给定经纬度点方向上以外一定距离的点坐标
    :param start_lat: 起始纬度点
    :param start_lon: 起始经度点
    :param end_lat: 结束纬度点
    :param end_lon: 结束经度点
    :param distance: 距离 默认1 单位：km
    :return:
    """

    direction = calcu_azimuth(start_lat, start_lon, end_lat, end_lon)
    distance = geopy.distance.geodesic(distance)

    end = geopy.Point(end_lat, end_lon)
    next = distance.destination(end, direction)
    next_point = (next.latitude, next.longitude)

    return next_point


def get_point_index(lons, lats, point_lon, point_lat):
    """
    得到所选坐标点的位置序列
    ----------
    param: lons: 经度数据
    param: lat: 纬度数据
    param: point_lon: 绘图坐标点经度
    param: point_lat: 绘图坐标点纬度
    return: 坐标的位置序列
    ----------
    """

    x = abs(lons - point_lon)
    y = abs(lats - point_lat)
    index = np.min(np.where(x == np.min(x)))
    indey = np.min(np.where(y == np.min(y)))

    return index, indey


def draw(lons, lats, data, pic_file, ele, distance=0, title=None, theta=360):
    """
    画图

    :param lons: 经度数组(lon,lat)
    :param lats: 纬度数组(lon,lat)
    :param data: 数据数组(lon,lat)
    :param pic_file: 要保存的图片本地地址
    :param ele: 要素名
    :param distance: 雷达观测范围
    :param title: 绘图标题
    :param theta: 添加额外的径向
    :return:
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 补全360-0的色块
    lons = np.row_stack((lons, lons[0]))
    lats = np.row_stack((lats, lats[0]))
    data = np.row_stack((data, data[0]))

    cmap = ListedColormap(colormap[ele]["colors"])
    norm = BoundaryNorm(boundaries=colormap[ele]["levels"], ncolors=len(colormap[ele]["levels"]))
    pic = plt.pcolormesh(lons, lats, data, norm=norm, cmap=cmap, shading='nearest')

    position = fig.add_axes([0.93, 0.1, 0.02, 0.78])
    if ele == 'scw':
        bins = [1, 2, 3]
        nbin = len(bins) - 1
        cmap = ListedColormap(['#a0a0a0', '#ff0000'])
        norm4 = mpl.colors.BoundaryNorm(bins, nbin)
        im4 = mpl.cm.ScalarMappable(norm=norm4, cmap=cmap)
        cb = plt.colorbar(im4, cax=position)
    else:
        cb = plt.colorbar(cax=position, ticks=colormap[ele].get('levels'), shrink=0.9, pad=0.035)
        cb.ax.set_title(colormap[ele]['label'])  # 设置单位

    if ele == 'hcl':
        # br = plt.colorbar(cax=position, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        cb.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        cb.set_ticklabels(['地物', '湍流', '干雪', '湿雪', '冰晶', '霰', '大滴', '雨', '大雨', '雹'])
    elif ele == 'scw':
        cb.set_ticks([1.5, 2.5])
        cb.set_ticklabels(['非过冷水', '过冷水'])
    elif ele == 'ccl':
        cb.set_ticks([0.5, 1.5])
        cb.set_ticklabels(['对流云', '层状云'])

    ax.axis('off')

    # ============================添加极坐标线=================================
    ax1 = fig.add_axes([0.13, 0.111, 0.767, 0.767], polar=True)
    # angle_num = 4  # 将圆等份
    # ax1.set_thetagrids(range(0, 360, int(360 / angle_num)), labels=[""] * angle_num)  # 设置角度刻度标签（这里角度标签labels都为空）
    if theta < 90:
        theta_n = 90 - theta
    else:
        theta_n = 450 - theta
    thetas = [0, 90, 180, 270, theta_n]
    thetas.sort()
    ax1.set_thetagrids(thetas, labels=[""] * len(thetas))
    distance_list = [round(distance / 4000, -1), round(distance / 3000), round(distance / 1800), distance / 1000]
    ax1.set_rgrids(distance_list, labels=["{}km".format(int(i)) for i in distance_list])  # 设置距离标签（这里刻度标签labels都为空）
    # =======================================================================================
    plt.title(title, fontsize=15)

    plt.savefig(pic_file, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    # 上传图片
    # minio_client.put_object(pic_file)


class QCReportDraw:
    def __init__(self, fileinfo, jobid, ele=None, img_type=None):
        self.jobid = jobid
        self.ele = ele
        self.img_type = img_type
        ea = Effect_Analysis(fileinfo, jobid, ele, '质控报告', 1)
        path = config_info.get_eff_cfg['input_dir_polar']
        self.polar_input_dir = ea.get_file_path(path)

        self.output_path = config_info.get_qc_cfg['pic_path']

    def draw_azimuth(self, ele, azimuths=None, angle=None):
        """
        各层仰角绘图
        :param ele: 绘图要素
        :param azimuths: 数据仰角
        :param angle: 方位角
        :return:
        """
        pic_info = []
        if azimuths is None:
            azimuths = []
        with Dataset(self.polar_input_dir[0]) as ds:
            dim1 = ds.variables['Dim1'][:]
            dim2 = ds.variables['Dim2'][:]
            dim3 = ds.variables['Dim3'][:]
            longitude, latitude = ds.variables['RadLon'][:], ds.variables['RadLat'][:]
            radar_hgt = ds.variables['RadHgt'][:]
            _, _, lons, lats = get_lon_lat(dim3 / 1000, dim2, radar_hgt, longitude, latitude)
            if not azimuths:
                azimuths = [2, 5]
            else:
                try:
                    ids = []
                    for i in azimuths:
                        ids.append(np.argwhere(dim1 == i)[0][0])
                    azimuths = ids
                except IndexError:
                    raise IndexError("仰角参数错误")
            for azimuth in azimuths:
                f_time = datetime.fromtimestamp(ds.RadTime)

                ele_data = ds.variables[ele][azimuth, :, :] if ele != 'scw' else ds.variables['hcl'][azimuth, :, :]
                ele_data[ele_data <= -999] = np.nan

                title = f'时间: {f_time.strftime("%Y-%m-%d %H:%M:%S")}    站号: {ds.TaskName}    仰角: {dim1[azimuth]}°'

                s_time = f_time.strftime('%Y%m%d%H%M%S')
                pic_file = os.path.join(self.output_path, s_time[0:4], s_time[4:6], s_time[6:8], str(self.jobid))
                if not Path(pic_file).exists():
                    Path.mkdir(Path(pic_file), parents=True, exist_ok=True)
                pic_path = os.path.join(pic_file, f"{f_time.strftime('%Y%m%d%H%M%S')}_{ele}_{dim1[azimuth]}.png")
                draw(lons, lats, ele_data, pic_path, ele, int(np.max(dim3)), title, angle)

                pic_info.append(
                    {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                     "element": f'{ele}_elevation'})

        return pic_info

    def draw_radial_line(self, ele, azimuths, angle):
        """
        径向折线图分析
        :param ele: 绘图要素
        :param azimuths: 数据仰角
        :param angle: 方位角
        :return:
        """
        pic_info = []
        with Dataset(self.polar_input_dir[0]) as ds:
            f_time = datetime.fromtimestamp(ds.RadTime)

            dim1 = ds.variables['Dim1'][:]
            dim2 = ds.variables['Dim2'][:]
            dim3 = ds.variables['Dim3'][:]
            idx = np.argmin(np.abs(np.array(dim2) - angle))

            if not azimuths:
                azimuths = [2, 5]
            else:
                try:
                    ids = []
                    for i in azimuths:
                        ids.append(np.argwhere(dim1 == i)[0][0])
                    azimuths = ids
                except IndexError:
                    raise IndexError("仰角参数错误")
            for azimuth in azimuths:
                # 取特定仰角、方位角的数据
                data = ds.variables[ele][azimuth, idx, :]
                x = dim3.reshape(len(dim3), 1) / 1000
                fig, ax = plt.subplots(figsize=(7, 2))
                plt.plot(x, data)
                plt.ylabel(f'{colormap[ele]["title"]}({colormap[ele]["label"]})')
                plt.xlabel(f'距离(km)')
                plt.xlim([0, dim3[-1] / 1000])
                title = f'时间: {f_time.strftime("%Y-%m-%d %H:%M:%S")}    站号: {ds.TaskName}    仰角: {dim1[azimuth]}°'
                plt.title(title)  # , fontsize=10

                s_time = f_time.strftime('%Y%m%d%H%M%S')
                pic_file = os.path.join(self.output_path, s_time[0:4], s_time[4:6], s_time[6:8], str(self.jobid))
                if not Path(pic_file).exists():
                    Path.mkdir(Path(pic_file), parents=True, exist_ok=True)
                pic_path = os.path.join(pic_file,
                                        f"{f_time.strftime('%Y%m%d%H%M%S')}_{ele}_{dim1[azimuth]:.3f}_{angle:.3f}.png")
                plt.savefig(pic_path, dpi=500, bbox_inches="tight", pad_inches=0.05)
                plt.close()
            # 上传图片到minio
                # minio_client.put_object(pic_path)

                pic_info.append({"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                                 "element": f'{ele}_line'})

        return pic_info

    def draw_scatter(self, ele_x, ele_y, azimuths=None):
        """
        散点图分析
        :param ele_x: 绘图要素
        :param ele_y: 绘图要素
        :param azimuths: 数据仰角
        :return:
        """
        pic_info = []
        with Dataset(self.polar_input_dir[0]) as ds:
            f_time = datetime.fromtimestamp(ds.RadTime)
            dim1 = ds.variables['Dim1'][:]
            dim2 = ds.variables['Dim2'][:]
            dim3 = ds.variables['Dim3'][:]

            if not azimuths:
                azimuths = [2, 5]
            else:
                try:
                    ids = []
                    for i in azimuths:
                        ids.append(np.argwhere(dim1 == i)[0][0])
                    azimuths = ids
                except IndexError:
                    raise IndexError("仰角参数错误")
            for azimuth in azimuths:
                x = ds.variables[ele_x][azimuth, :, :]
                y = ds.variables[ele_y][azimuth, :, :]
                y[y <= -997] = np.nan
                fig, ax = plt.subplots(figsize=(7, 2))
                plt.scatter(x, y, s=0.5)
                plt.ylabel(f'{colormap[ele_y]["title"]}({colormap[ele_y]["label"]})')
                plt.xlabel(f'{colormap[ele_x]["title"]}({colormap[ele_x]["label"]})')
                if ele_x == 'snr':
                    plt.xlim([0, 55])
                else:
                    plt.xlim([-10, 55])
                if ele_y == 'zdr':
                    plt.axhline(y=np.nanmean(y), xmin=0, xmax=1, linestyle='-', color='r', linewidth=1)
                    plt.text(55, np.nanmean(y) + 0.1, f'{np.nanmean(y): .2f}', color='r')
                plt.title(f'时间: {f_time.strftime("%Y-%m-%d %H:%M:%S")}    站点: {ds.TaskName}    仰角: {dim1[azimuth]}°')

                s_time = f_time.strftime('%Y%m%d%H%M%S')
                pic_file = os.path.join(self.output_path, s_time[0:4], s_time[4:6], s_time[6:8], str(self.jobid))
                if not Path(pic_file).exists():
                    Path.mkdir(Path(pic_file), parents=True, exist_ok=True)
                pic_path = os.path.join(pic_file,
                                        f"{f_time.strftime('%Y%m%d%H%M%S')}_{ele_x}_{ele_y}_{dim1[azimuth]:.3f}.png")
                plt.savefig(pic_path, dpi=500, transparent=False, bbox_inches="tight", pad_inches=0.05)
                plt.close()
                logger.info(f"生成图片：{pic_path}")
                # 上传图片到minio
                # minio_client.put_object(pic_path)
                pic_info.append(
                    {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                     "element": f'{ele_y}_scatter'})

        return pic_info

    def draw_mds(self, azimuths=None):
        """
        回波强度 雷达最小可辨信号分析
        :param azimuths: 数据仰角
        :return:
        """
        pic_info = []
        with Dataset(self.polar_input_dir[0], 'r') as ds:
            f_time = datetime.fromtimestamp(ds.RadTime)
            dim3 = ds.dimensions['Dim3'].size
            exit_d = np.zeros([dim3, 76])  # 库数与基本反射率分组数
            dim1 = ds.variables['Dim1'][:]
            if not azimuths:
                azimuths = [2, 5]
            else:
                try:
                    ids = []
                    for i in azimuths:
                        ids.append(np.argwhere(dim1 == i)[0][0])
                    azimuths = ids
                except IndexError:
                    raise IndexError("仰角参数错误")
        for azimuth in azimuths:
            for f in self.polar_input_dir:
                with Dataset(f, 'r') as ds:
                    f_time_d = datetime.fromtimestamp(ds.RadTime)
                    data = ds.variables["ref"][azimuth, :, :]
                    ll = []
                    for i in range(data.shape[-1]):
                        num_l = []
                        for j in np.arange(0, 76):
                            if j == 0:
                                num_l.append(len(np.argwhere(data[:, i] <= j)))
                            else:
                                num_l.append(len(np.argwhere(data[:, i] <= j)) - len(np.argwhere(data[:, i] <= j - 1)))
                        ll.append(num_l)
                    dd = np.array(ll, dtype=float)
                    exit_d = dd + exit_d

            exit_d[exit_d == 0] = None
            tick = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210]
            cmap = ListedColormap(colormap['ref']["colors"])
            norm = BoundaryNorm(boundaries=tick, ncolors=len(colormap["ref"]["levels"]))
            fig, ax = plt.subplots()
            y = np.linspace(0, 75, 76)
            x = np.linspace(0.030, dim3 * 0.030, dim3)
            im = ax.pcolormesh(x, y, exit_d.T, cmap=cmap, norm=norm)
            cb = fig.colorbar(im, ticks=tick, shrink=0.9, pad=0.02, extend="max")

            y = []
            for j in range(dim3):
                position = np.argwhere(~np.isnan(exit_d))
                try:
                    ins = np.min(position[position[:, 0] == j])
                except ValueError:
                    ins = 0
                y.append(ins)
            z1 = np.polyfit(x[0:-20], y[0:-20], 2)  # 用3次多项式拟合，输出系数从高到0
            p1 = np.poly1d(z1)  # 使用次数合成多项式
            y_pre = p1(x)
            plt.plot(x, y_pre, c='k')

            ax.set(xlabel='距离(km)', ylabel='基本反射率(dBZ)')
            plt.title(f'MDS    仰角: {dim1[azimuth]}°')

            s_time = f_time.strftime('%Y%m%d%H%M%S')
            pic_file = os.path.join(self.output_path, s_time[0:4], s_time[4:6], s_time[6:8], str(self.jobid))
            if not Path(pic_file).exists():
                Path.mkdir(Path(pic_file), parents=True, exist_ok=True)
            pic_path = os.path.join(pic_file,
                                    f"{f_time.strftime('%Y%m%d%H%M%S')}_{f_time_d.strftime('%Y%m%d%H%M%S')}_{dim1[azimuth]:.3f}.png")
            plt.savefig(pic_path, dpi=500, transparent=False, bbox_inches="tight", pad_inches=0.05)
            plt.close()

            logger.info(f"生成图片：{pic_path}")
            # 上传图片到minio
            # minio_client.put_object(pic_path)
            pic_info.append({"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                             "element": f'ref_mds'})

        return pic_info

    def run(self, azimuth=None, angle=245):
        pic_infos = []
        if azimuth:
            if type(azimuth) is not list:
                azimuth = [azimuth]
        # if self.ele and self.img_type:
        #     if self.img_type == 'elevation':
        #         pic_infos.extend(self.draw_azimuth(self.ele, azimuth, angle))
        #     elif self.img_type == 'line':
        #         pic_infos.extend(self.draw_radial_line(self.ele, azimuth, angle))
        #     elif self.img_type == 'scatter':
        #         pic_infos.extend(self.draw_scatter(self.ele[0], self.ele[1]))
        #     elif self.img_type == 'mds':
        #         pic_infos.extend(self.draw_mds(azimuth))

        for ele in eles_pol:
            pic_infos.extend(self.draw_azimuth(ele, azimuth, angle))
        for ele in ['ref', 'kdp', 'pdp']:
            pic_infos.extend(self.draw_radial_line(ele, azimuth, angle))
        for eles in [['ref', 'zdr'], ['snr', 'rhv']]:
            pic_infos.extend(self.draw_scatter(eles[0], eles[1], azimuth))
        pic_infos.extend(self.draw_mds(azimuth))

        return [{"picFiles": pic_infos}]


if __name__ == '__main__':
    output_path = r'D:\gzfb\贵州防雹文档资料\质控报告'
    ele = "ref"
    QCReportDraw('D:', 33, 'cross', 1).run(azimuth=[2.25, 4.5], angle=234.46)
