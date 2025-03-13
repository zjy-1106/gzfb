#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
获取雷达剖面图
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
import re
import glob
import itertools

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from netCDF4 import Dataset
from metpy.interpolate import cross_section
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from geopy.distance import geodesic

from service.colormap import colormap
from config.config import config_info
from service.utils import transfer_path, get_limit_index, extend_point, get_file_path, add_right_cax
# from utils.file_uploader import minio_client

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

mpl.rc("font", family='DengXian')
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8
FONTSIZE = 12
eles_pol = ['ref', 'vel', 'wid', 'zdr', 'rhv', 'pdp', 'kdp', 'snr', 'hcl', 'ccl', 'scw']
eles_grb = ['ref', 'zdr', 'rhv', 'kdp', 'hcl', 'ccl', 'scw']


class Section:
    def __init__(self, filelist, ele, img_type, jobid, title=None, elements=None):
        self.layer = None
        # 剖面位置
        self.ele = ele
        self.eles = None
        self.img_type = img_type
        self.title = title
        path_cfg = config_info.get_radar_section_config
        if elements:
            if img_type == "radial":
                self.nc_files = get_file_path(filelist, path_cfg['polar_input_dir'], img_type=img_type)
            else:
                self.nc_files = get_file_path(filelist, path_cfg['input_dir'], img_type=img_type)
            self.eles = [el.split('_')[0] for el in elements]
        else:
            self.fileid = [str(fileinfo['fileId']) for fileinfo in filelist]
            if filelist[0].get('radType') == 'HTM':
                time_str = re.search(r'\d{8}_\d{2}', os.path.basename(filelist[0].get('filePath'))).group().replace('_',
                                                                                                                    '')
                time_str = (datetime.strptime(time_str, '%Y%m%d%H') - timedelta(hours=8)).strftime('%Y%m%d')
            else:
                time_str = re.search(r'\d{8}', os.path.basename(filelist[0].get('filePath'))).group()  # 源文件为世界时

            self.input_dir = os.path.join(path_cfg['input_dir'], time_str)
            self.polar_input_dir = os.path.join(path_cfg['polar_input_dir'], time_str, str(self.fileid[0]))

        self.output_path = os.path.join(path_cfg['pic_path'], str(jobid))
        if not Path(self.output_path).exists():
            Path.mkdir(Path(self.output_path), parents=True, exist_ok=True)

    def draw_cross_section(self, start_lat, end_lat, start_lon, end_lon, blast_point_list=None, center_point=None):
        # 绘制任意两点剖面
        # nc_file = glob.glob(os.path.join(self.input_dir, '-'.join(self.fileid), '*'))  # 输入nc文件
        # if not nc_file:
        #     nc_file = glob.glob(os.path.join(self.input_dir, '-'.join(self.fileid[::-1]), '*'))

        inp = list(itertools.product(self.fileid, repeat=len(self.fileid)))
        for id_list in inp:
            try:
                nc_file = ''.join(
                    glob.glob(os.path.join(self.input_dir, '-'.join(id_list), "*"))[0])  # 文件路径
                fi = True
            except IndexError as e:
                fi = False
                continue
            else:
                break
        if fi == False:
            raise Exception(f'文件未查询成功')
        # with xr.open_dataset(self.file) as ds:
        ele = self.ele
        if self.ele == 'scw':  # scw读取hcl
            ele = 'hcl'
        # try:
        #     nc_file = nc_file[0]
        # except IndexError:
        #     raise Exception("格点nc文件不存在")
        pic_info = []
        with Dataset(nc_file, 'r') as ds:

            if self.ele:
                eles = [self.ele.split('_')[0]]
            else:  # 不传要素则绘制全部
                eles = eles_grb
            for ele in eles:

                data = ds.variables[ele][:] if ele != 'scw' else ds.variables['hcl'][:]  # 需要绘制的数据
                FLat = ds.variables['Dim2'][:]
                FLon = ds.variables['Dim3'][:]
                self.layer = np.array(ds.variables['Dim1'][:])  # 数据层数
                data_time = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y-%m-%d %H:%M')

                if FLat.ndim == 1:
                    xdd = xr.Dataset(data_vars={ele: (['height', 'lat', 'lon'], data)},
                                     coords={'lon': (['lon'], FLon),
                                             'lat': (['lat'], FLat),
                                             'height': (['height'], self.layer)})
                else:
                    lat = np.linspace(np.min(FLat), np.max(FLat), 361)[::-1]
                    lon = np.linspace(np.min(FLon), np.max(FLon), 1000)
                    xdd = xr.Dataset(data_vars={ele: (['height', 'lat', 'lon'], data)},
                                     coords={'lon': (['lon'], lon),
                                             'lat': (['lat'], lat),
                                             'height': (['height'], self.layer)})

                # 若是传入爆炸点，则取爆炸点前后1km的点进行剖面分析
                extend_for = (start_lat, start_lon)
                extend_back = (end_lat, end_lon)
                if blast_point_list is not None and len(blast_point_list) > 1:
                    extend_for = extend_point(end_lat, end_lon, start_lat, start_lon)
                    extend_back = extend_point(start_lat, start_lon, end_lat, end_lon)
                elif blast_point_list is not None and len(blast_point_list) == 1:
                    extend_back = extend_point(start_lat, start_lon, end_lat, end_lon)
                xdd = xdd.metpy.parse_cf().squeeze()
                data = cross_section(xdd, extend_for, extend_back)

                # %%
                x, y, z = data['index'], data['height'], data[ele]
                inter = get_limit_index(len(x))

                fig = plt.figure(figsize=(15, 7))
                ax = fig.add_axes([0.09, 0.1, 0.85, 0.8])
                cmap = ListedColormap(colormap[ele]["colors"])
                norm = BoundaryNorm(boundaries=colormap[ele]["levels"], ncolors=len(colormap[ele]["levels"]))
                im = ax.pcolormesh(x, y / 1000, z, norm=norm, cmap=cmap, shading='auto')
                r = geodesic((start_lat, start_lon), (end_lat, end_lon))  # 爆炸点间离
                # ----添加爆炸点---------------
                if blast_point_list:
                    station = int((1 - (r / geodesic(extend_for, extend_back))) * 100)  # 爆炸点位置
                    blast_height = blast_point_list[0][-1]
                    if len(blast_point_list) > 1:
                        blast_height = blast_point_list[-1][-1]
                        station = int(station / 2)
                        ax.scatter(x[station], blast_height / 1000, s=50, c='none', marker="^", edgecolors='k',
                                   linewidths=1.5)  # 空心填色c=none,
                    ax.scatter(x[-station], blast_height / 1000, s=50, c='none', marker="^", edgecolors='k',
                               linewidths=1.5)
                    ax.set_xlabel("距离(km)", fontsize=20)
                    r = geodesic(extend_for, extend_back)
                # ---------------------------
                # ----添加质心点---------------
                if center_point:
                    station = int((geodesic(extend_for, center_point[0:2][::-1]) / r) * 100)  # 爆炸点位置
                    ax.scatter(x[station], center_point[-1] / 1000, s=50, c='none', marker="^", edgecolors='k',
                               linewidths=1.5)
                # ---------------------------
                ax.set_ylabel("高度(km)", fontsize=20)
                if ele == 'scw':
                    bins = [1, 2, 3]
                    nbin = len(bins) - 1
                    cmap = ListedColormap(['#a0a0a0', '#ff0000'])
                    norm4 = mcolors.BoundaryNorm(bins, nbin)
                    im4 = cm.ScalarMappable(norm=norm4, cmap=cmap)
                    cb = fig.colorbar(im4, ax=ax)
                else:
                    cb = fig.colorbar(im, ticks=colormap[ele].get('levels'), shrink=0.9, pad=0.035)
                    cb.ax.set_title(colormap[ele]['label'], fontsize=14)  # 设置单位

                # x轴坐标
                ticks = data['index'].data[::inter]
                ys = np.round(data['lon'].data[::inter], 2)
                xs = np.round(data['lat'].data[::inter], 2)
                labels = []

                r_lab = np.round(np.linspace(0, r.km, 100)[::inter], 2)
                for n, y_val in enumerate(ys):
                    lo = f'{abs(y_val)}°E' if y_val > 0 else f'{abs(y_val)}°W'
                    la = f'{xs[n]}°N' if xs[n] > 0 else f'{xs[n]}°S'
                    if blast_point_list:
                        labels.append(str(r_lab[n]))
                    else:
                        labels.append(f'{lo}\n{la}')
                if self.ele:
                    title = self.title + '  ' + data_time
                else:
                    title = self.title + '  ' + colormap[ele]['title'] + '  ' + data_time
                plt.title(title, fontsize=22)
                plt.xticks(ticks, labels, fontsize=15)
                plt.yticks(fontsize=20)
                plt.ylim([2.5, y[-1] / 1000])
                lim = get_limit_index(len(y), 6)
                # plt.yticks(y[::lim], [f'{int(i / 1000)}' for i in self.layer[::lim]])
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
                cb.ax.tick_params(labelsize=14)

                time_str = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y%m%d%H%M%S')
                pic_path = os.path.join(self.output_path, f"{time_str}_{ele}.png")
                plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05, dpi=400)
                plt.close()
                # 上传图片到minio
                # minio_client.put_object(pic_path)

                pic_info.append(
                    {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                     "element": f'{ele}_pro'})
        return [{"picFiles": pic_info}]

    def draw_radial_section(self, angle, blast_height=None, blast_lon=None, blast_lat=None):
        # 绘制径向剖面
        nc_file = os.path.join(self.polar_input_dir, os.listdir(self.polar_input_dir)[0])
        pic_info = []
        plt.rcParams['font.size'] = 10
        with Dataset(nc_file) as ds:
            dim1 = ds.variables['Dim1'][:]
            dim2 = ds.variables['Dim2'][:]
            dim3 = ds.variables['Dim3'][:]
            deg2rad = np.pi / 180
            idx = np.argmin(np.abs(np.array(dim2) - angle))

            if self.ele:
                eles = [self.ele.split('_')[0]]
            else:  # 不传要素则绘制全部
                eles = eles_pol
            for ele in eles:
                # if ele == 'scw':
                #     ele = 'hcl'

                # 取特定相位角的数据
                data = ds.variables[ele][:, idx, :] if ele != 'scw' else ds.variables['hcl'][:, idx, :]
                cmap = ListedColormap(colormap[ele].get('colors'))
                norm = BoundaryNorm(boundaries=colormap[ele].get('levels'),
                                    ncolors=len(colormap[ele].get('levels')))
                x = dim3.reshape(len(dim3), 1) * np.cos(dim1.reshape(1, len(dim1)) * deg2rad) / 1000
                y = dim3.reshape(len(dim3), 1) * np.sin(dim1.reshape(1, len(dim1)) * deg2rad) / 1000
                fig, ax = plt.subplots()
                ax.set(xlabel='距离(km)', ylabel='高度(km)')
                data_time = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y-%m-%d %H:%M')
                if self.ele:
                    title = self.title + '  ' + data_time
                else:
                    title = self.title + '  ' + colormap[ele]['title'] + '  ' + data_time
                plt.title(title)

                ele_data = data.T
                ele_data = np.array(ele_data)
                ele_data[ele_data == -999] = None
                # ------返回数据---------------------
                # data_temp = []
                # for da in ele_data:
                #     data_temp.append(list(da))
                # ----------------------------------
                im = ax.pcolormesh(x, y, ele_data, cmap=cmap, norm=norm, shading='nearest')

                # ----添加爆炸点---------------
                if blast_height is not None and not np.isnan(blast_height):
                    h_t = blast_height
                    h_t = h_t - ds.variables['RadHgt'][0]
                    radlon = ds.variables['RadLon'][0]
                    radlat = ds.variables['RadLat'][0]
                    r = geodesic((radlat, radlon), (blast_lat, blast_lon))
                    ax.scatter(r.km, h_t / 1000, s=50, c='none', marker="^", edgecolors='k',
                               linewidths=1.5)  # 空心填色c=none,
                # ---------------------------
                d1, d2 = np.where(~np.isnan(ele_data))
                x_max = 45
                y_max = 20
                if d1.size and d2.size:
                    x_max = int((np.max(dim3[d1] * np.cos(np.array(dim1[d2]) * deg2rad))) / 1000) + 1
                    y_max = int((np.max(dim3[d1] * np.sin(np.array(dim1[d2]) * deg2rad))) / 1000) + 1
                    y_max = 20  # 固定展示到20km
                ax.set(xlim=(0, x_max), ylim=(0, y_max))

                if ele == 'scw':
                    bins = [1, 2, 3]
                    nbin = len(bins) - 1
                    cmap = ListedColormap(['#a0a0a0', '#ff0000'])
                    norm4 = mcolors.BoundaryNorm(bins, nbin)
                    im4 = cm.ScalarMappable(norm=norm4, cmap=cmap)
                    cb = fig.colorbar(im4, ax=ax)
                else:
                    cb = fig.colorbar(im, ticks=colormap[ele].get('levels'), shrink=0.9, pad=0.02)
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

                time_str = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y%m%d%H%M%S')
                pic_path = os.path.join(self.output_path, f"{time_str}_{ele}.png")
                plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
                plt.close()
                # 上传图片到minio
                # minio_client.put_object(pic_path)

                pic_info.append(
                    {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                     "element": f'{ele}_pro',
                     "otherData": {"xLimit": [0, x_max], "yLimit": [0, y_max],
                                   "maxThreshold": float(np.nanmax(ele_data))},
                     "unit": 'km'})
        return [{"picFiles": pic_info}]

    def draw_radial_section_sum(self, angle, blast_height=None, blast_lon=None, blast_lat=None, eff_time=None):
        eff_times = [datetime.strptime(eff_t, '%Y%m%d%H%M%S') for eff_t in eff_time[0]]
        plt.rcParams['font.size'] = 15
        # eles = ['ref', 'vel', 'wid']
        nrows, ncols = len(self.eles), len(self.nc_files)
        pic_info = []

        figsize = (6 * ncols, 5 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=400, sharex="all", sharey="all")
        for ele_idx, ele in enumerate(self.eles):
            for time_idx, nc_file in enumerate(self.nc_files):
                with Dataset(nc_file) as ds:
                    dim1 = ds.variables['Dim1'][:]
                    dim2 = ds.variables['Dim2'][:]
                    dim3 = ds.variables['Dim3'][:]
                    deg2rad = np.pi / 180
                    idx = np.argmin(np.abs(np.array(dim2) - angle))
                    time_str = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y%m%d%H%M%S')

                    # 取特定相位角的数据
                    data = ds.variables[ele][:, idx, :] if ele != 'scw' else ds.variables['hcl'][:, idx, :]
                    cmap = ListedColormap(colormap[ele].get('colors'))
                    norm = BoundaryNorm(boundaries=colormap[ele].get('levels'),
                                        ncolors=len(colormap[ele].get('levels')))
                    x = dim3.reshape(len(dim3), 1) * np.cos(dim1.reshape(1, len(dim1)) * deg2rad) / 1000
                    y = dim3.reshape(len(dim3), 1) * np.sin(dim1.reshape(1, len(dim1)) * deg2rad) / 1000
                    if ncols == 1 and nrows == 1:
                        ax = axs
                    elif ncols == 1:
                        ax = axs[ele_idx]
                    elif nrows == 1:
                        ax = axs[time_idx]
                    else:
                        ax = axs[ele_idx, time_idx]
                    data_time = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y-%m-%d %H:%M')
                    if datetime.fromtimestamp(ds.getncattr('RadTime')) > eff_times[-1]:
                        title = '作业后\t' + data_time
                    elif datetime.fromtimestamp(ds.getncattr('RadTime')) < eff_times[0]:
                        title = '作业前\t' + data_time
                    else:
                        title = '作业中\t' + data_time

                    ele_data = data.T
                    ele_data = np.array(ele_data)
                    ele_data[ele_data == -999] = None
                    # ------返回数据---------------------
                    # data_temp = []
                    # for da in ele_data:
                    #     data_temp.append(list(da))
                    # ----------------------------------
                    im = ax.pcolormesh(x, y, ele_data, cmap=cmap, norm=norm, shading='nearest')

                    # ----添加爆炸点---------------
                    if blast_height is not None and not np.isnan(blast_height):
                        h_t = blast_height
                        h_t = h_t - ds.variables['RadHgt'][0]
                        radlon = ds.variables['RadLon'][0]
                        radlat = ds.variables['RadLat'][0]
                        r = geodesic((radlat, radlon), (blast_lat, blast_lon))
                        ax.scatter(r.km, h_t / 1000, s=50, c='none', marker="^", edgecolors='k',
                                   linewidths=1.5)  # 空心填色c=none,
                    # ---------------------------
                    d1, d2 = np.where(~np.isnan(ele_data))
                    x_max = 45
                    y_max = 20
                    if d1.size and d2.size:
                        x_max = int((np.max(dim3[d1] * np.cos(np.array(dim1[d2]) * deg2rad))) / 1000) + 1
                        # y_max = int((np.max(dim3[d1] * np.sin(np.array(dim1[d2]) * deg2rad))) / 1000) + 1
                        y_max = 20  # 固定展示到20km
                    ax.set(xlim=(0, x_max), ylim=(0, y_max))
                    if ele_idx == 0:
                        ax.set_title(title)
                    if ele_idx == nrows - 1:
                        ax.set(xlabel='距离(km)')
                    if time_idx == 0:
                        ax.set(ylabel='高度(km)')
            if ele == 'scw':
                bins = [1, 2, 3]
                nbin = len(bins) - 1
                cmap = ListedColormap(['#a0a0a0', '#ff0000'])
                norm4 = mcolors.BoundaryNorm(bins, nbin)
                im4 = cm.ScalarMappable(norm=norm4, cmap=cmap)
                cax_0 = add_right_cax(ax, pad=0.03 / ncols, width=0.03 / ncols)
                cb = fig.colorbar(im4, ax=ax, cax=cax_0)
            else:
                cax_0 = add_right_cax(ax, pad=0.03 / ncols, width=0.03 / ncols)
                cb = fig.colorbar(im, ax=ax, cax=cax_0, ticks=colormap[ele].get('levels'))

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
            else:
                cb.ax.set_ylabel(f"{ele} ({colormap[ele]['label']})")  # 设置单位
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        pic_path = os.path.join(self.output_path, f"{time_str}_{self.eles[0]}_{nrows * ncols}.png")
        plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        # 上传图片到minio
        # minio_client.put_object(pic_path)

        pic_info.append(
            {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
             "element": f'{self.eles[0]}_pro',
             "otherData": {"xLimit": [0, x_max], "yLimit": [0, y_max],
                           "maxThreshold": float(np.nanmax(ele_data))},
             "unit": 'km'})

        return [{"picFiles": pic_info}]

    def draw_cross_section_sum(self, start_lat, end_lat, start_lon, end_lon, blast_point_list=None, center_point=None,
                               eff_time=None):
        eff_times = [datetime.strptime(eff_t, '%Y%m%d%H%M%S') for eff_t in eff_time[0]]
        plt.rcParams['font.size'] = 15
        # eles = ['ref', 'vel', 'wid']
        nrows, ncols = len(self.eles), len(self.nc_files)
        pic_info = []

        figsize = (6 * ncols, 5 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=400, sharex="all", sharey="all")
        for ele_idx, ele in enumerate(self.eles):
            for time_idx, nc_file in enumerate(self.nc_files):
                with Dataset(nc_file) as ds:
                    data = ds.variables[ele][:] if ele != 'scw' else ds.variables['hcl'][:]  # 需要绘制的数据
                    FLat = ds.variables['Dim2'][:]
                    FLon = ds.variables['Dim3'][:]
                    self.layer = np.array(ds.variables['Dim1'][:])  # 数据层数
                    data_time = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y-%m-%d %H:%M')
                    time_str = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y%m%d%H%M%S')

                    if FLat.ndim == 1:
                        xdd = xr.Dataset(data_vars={ele: (['height', 'lat', 'lon'], data)},
                                         coords={'lon': (['lon'], FLon),
                                                 'lat': (['lat'], FLat),
                                                 'height': (['height'], self.layer)})
                    else:
                        lat = np.linspace(np.min(FLat), np.max(FLat), 361)[::-1]
                        lon = np.linspace(np.min(FLon), np.max(FLon), 1000)
                        xdd = xr.Dataset(data_vars={ele: (['height', 'lat', 'lon'], data)},
                                         coords={'lon': (['lon'], lon),
                                                 'lat': (['lat'], lat),
                                                 'height': (['height'], self.layer)})

                    # 若是传入爆炸点，则取爆炸点前后1km的点进行剖面分析
                    extend_for = (start_lat, start_lon)
                    extend_back = (end_lat, end_lon)
                    if blast_point_list is not None and len(blast_point_list) > 1:
                        extend_for = extend_point(end_lat, end_lon, start_lat, start_lon)
                        extend_back = extend_point(start_lat, start_lon, end_lat, end_lon)
                    elif blast_point_list is not None and len(blast_point_list) == 1:
                        extend_back = extend_point(start_lat, start_lon, end_lat, end_lon)
                    xdd = xdd.metpy.parse_cf().squeeze()
                    data = cross_section(xdd, extend_for, extend_back)

                    # %%
                    if ncols == 1 and nrows == 1:
                        ax = axs
                    elif ncols == 1:
                        ax = axs[ele_idx]
                    elif nrows == 1:
                        ax = axs[time_idx]
                    else:
                        ax = axs[ele_idx, time_idx]

                    x, y, z = data['index'], data['height'], data[ele]
                    inter = 15  # get_limit_index(len(x))

                    cmap = ListedColormap(colormap[ele]["colors"])
                    norm = BoundaryNorm(boundaries=colormap[ele]["levels"], ncolors=len(colormap[ele]["levels"]))
                    im = ax.pcolormesh(x, y / 1000, z, norm=norm, cmap=cmap, shading='auto')
                    r = geodesic((start_lat, start_lon), (end_lat, end_lon))  # 爆炸点间离
                    # ----添加爆炸点---------------
                    if blast_point_list:
                        station = int((1 - (r / geodesic(extend_for, extend_back))) * 100)  # 爆炸点位置
                        blast_height = blast_point_list[0][-1]
                        if len(blast_point_list) > 1:
                            blast_height = blast_point_list[-1][-1]
                            station = int(station / 2)
                            ax.scatter(x[station], blast_height / 1000, s=50, c='none', marker="^", edgecolors='k',
                                       linewidths=1.5)  # 空心填色c=none,
                        ax.scatter(x[-station], blast_height / 1000, s=50, c='none', marker="^", edgecolors='k',
                                   linewidths=1.5)
                        r = geodesic(extend_for, extend_back)
                    # ---------------------------
                    # ----添加质心点---------------
                    if center_point:
                        station = int((geodesic(extend_for, center_point[0:2][::-1]) / r) * 100)  # 爆炸点位置
                        ax.scatter(x[station], center_point[-1] / 1000, s=50, c='none', marker="^", edgecolors='k',
                                   linewidths=1.5)
                    # ---------------------------
                    ax.set_ylim([2.5, y[-1] / 1000])
                    if datetime.fromtimestamp(ds.getncattr('RadTime')) > eff_times[-1]:
                        title = '作业后\t' + data_time
                    elif datetime.fromtimestamp(ds.getncattr('RadTime')) < eff_times[0]:
                        title = '作业前\t' + data_time
                    else:
                        title = '作业中\t' + data_time
                    if ele_idx == 0:
                        ax.set_title(title)
                    if ele_idx == nrows - 1:
                        ticks = data['index'].data[::inter]
                        ys = np.round(data['lon'].data[::inter], 2)
                        xs = np.round(data['lat'].data[::inter], 2)
                        labels = []

                        r_lab = np.round(np.linspace(0, r.km, 100)[::inter], 2)
                        for n, y_val in enumerate(ys):
                            lo = f'{abs(y_val)}°E' if y_val > 0 else f'{abs(y_val)}°W'
                            la = f'{xs[n]}°N' if xs[n] > 0 else f'{xs[n]}°S'
                            if blast_point_list:
                                labels.append(str(r_lab[n]))
                                ax.set(xlabel='距离(km)')
                            else:
                                labels.append(f'{lo}\n{la}')
                        ax.set_xticks(ticks)
                        ax.set_xticklabels(labels)
                    if time_idx == 0:
                        ax.set(ylabel='高度(km)')
            if ele == 'scw':
                bins = [1, 2, 3]
                nbin = len(bins) - 1
                cmap = ListedColormap(['#a0a0a0', '#ff0000'])
                norm4 = mcolors.BoundaryNorm(bins, nbin)
                im4 = cm.ScalarMappable(norm=norm4, cmap=cmap)
                cax_0 = add_right_cax(ax, pad=0.03 / ncols, width=0.03 / ncols)
                cb = fig.colorbar(im4, ax=ax, cax=cax_0)
            else:
                cax_0 = add_right_cax(ax, pad=0.03 / ncols, width=0.03 / ncols)
                cb = fig.colorbar(im, ax=ax, cax=cax_0, ticks=colormap[ele].get('levels'))

            if ele == 'hcl':
                cb.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                cb.set_ticklabels(['地物', '湍流', '干雪', '湿雪', '冰晶', '霰', '大滴', '雨', '大雨', '雹'])
            elif ele == 'scw':
                cb.set_ticks([1.5, 2.5])
                cb.set_ticklabels(['非过冷水', '过冷水'])
            elif ele == 'ccl':
                cb.set_ticks([0.5, 1.5])
                cb.set_ticklabels(['对流云', '层状云'])
            else:
                cb.ax.set_ylabel(f"{ele} ({colormap[ele]['label']})")  # 设置单位
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        pic_path = os.path.join(self.output_path, f"{time_str}_{self.eles[0]}_{nrows * ncols}.png")
        plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        # 上传图片到minio
        # minio_client.put_object(pic_path)

        pic_info.append(
            {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
             "element": f'{self.eles[0]}_pro',
             "otherData": None,
             "unit": 'km'})

        return [{"picFiles": pic_info}]

    def draw_radial_line(self, elevation, angle):
        """
        径向折线图分析
        :param elevation: 数据仰角
        :param angle: 方位角
        :return:
        """
        nc_file = os.path.join(self.polar_input_dir, os.listdir(self.polar_input_dir)[0])

        with Dataset(nc_file) as ds:
            dim1 = ds.variables['Dim1'][:]
            dim2 = ds.variables['Dim2'][:]
            dim3 = ds.variables['Dim3'][:]
            deg2rad = np.pi / 180
            idx = np.argmin(np.abs(np.array(dim2) - angle))
            idx_e = np.where(dim1 == elevation)[0][0]

            ele = self.ele
            if self.ele == 'scw':
                ele = 'hcl'

            # 取特定仰角、方位角的数据
            data = ds.variables[ele][idx_e, idx, :]
            x = dim3.reshape(len(dim3), 1) / 1000
            fig, ax = plt.subplots(figsize=(8, 4))
            plt.plot(x, data)
            plt.ylabel(f'{colormap[ele]["title"]}({colormap[ele]["label"]})')
            plt.xlabel(f'距离(km)')
            plt.xlim([0, dim3[-1] / 1000])
            plt.title(f'{self.title} 仰角: {elevation:.3f}° 方位角: {angle:.3f}°')  # , fontsize=10

            pic_path = os.path.join(self.output_path, f"{self.img_type}_{elevation:.3f}_{angle:.3f}_{self.ele}.png")
            plt.savefig(pic_path, bbox_inches="tight", pad_inches=0.05)
            plt.close()
            # 上传图片到minio
            # minio_client.put_object(pic_path)

            pic_info = {"filename": f"{self.ele}.png", "path": transfer_path(pic_path, is_win_path=True),
                        "element": self.ele}
        return [{"picFiles": [pic_info]}]

    def get_point_data(self, elevation, angle, distance):
        nc_file = os.path.join(self.polar_input_dir, os.listdir(self.polar_input_dir)[0])

        with Dataset(nc_file) as ds:
            dim1 = ds.variables['Dim1'][:]
            dim2 = ds.variables['Dim2'][:]
            dim3 = ds.variables['Dim3'][:]
            idx1 = np.argmin(np.abs(np.array(dim1) - elevation))
            idx2 = np.argmin(np.abs(np.array(dim2) - angle))
            idx3 = np.argmin(np.abs(np.array(dim3) - distance))
            data = float(ds.variables[self.ele][idx1, idx2, idx3])

        return [{"picFiles": [{
            "otherData": {
                "pointData": {
                    "value": data, "unit": colormap[self.ele].get('units')}
            }
        }]
        }]

    def run(self, start_lat=None, end_lat=None, start_lon=None, end_lon=None, angle=None,
            blast_point_list=None, elevation=None, distance=None, center_point=None, eff_time=None):
        if distance is None:
            if self.img_type == 'cross':
                if self.eles:
                    return self.draw_cross_section_sum(start_lat, end_lat, start_lon, end_lon,
                                                       blast_point_list=blast_point_list, center_point=center_point,
                                                       eff_time=eff_time)
                return self.draw_cross_section(start_lat, end_lat, start_lon, end_lon,
                                               blast_point_list=blast_point_list, center_point=center_point)
            elif self.img_type == 'analysis_line':
                return self.draw_radial_line(elevation, angle)
            else:
                if blast_point_list:
                    blast_lon, blast_lat, blast_h = blast_point_list[0]
                    if self.eles:
                        return self.draw_radial_section_sum(angle, blast_h, blast_lon, blast_lat, eff_time=eff_time)
                    return self.draw_radial_section(angle, blast_h, blast_lon, blast_lat)
                else:
                    if self.eles:
                        return self.draw_radial_section_sum(angle, eff_time=eff_time)
                    return self.draw_radial_section(angle)
        else:
            return self.get_point_data(elevation, angle, distance)
