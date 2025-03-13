#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
雨滴谱反演，质控后的雷达数据NC文件
estimate_rainrate.exe 为降水估测程序，
estimate_dsd.exe 为雨滴谱反演程序，
命令行参数依次为：可执行程序位置 质控后的雷达数据NC文件位置 生成的产品路径
"""
import glob
import os
import re

from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr
from matplotlib.dates import DateFormatter, rrulewrapper, RRuleLocator
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from loguru import logger
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

from mq.pubulisher import publish_temp_msg
from service.effect_analysis import set_x_time
from service.utils import get_lon_lat, transfer_path, get_point_data, temp_message, select_data, file_exit, \
     ax_add_time_range
from config.config import config_info
# from utils.file_uploader import minio_client

mpl.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
mpl.use('Agg')
FONTSIZE = 10

str_point = [103.01042080, 25.960774]  # 叠图区域
end_point = [104.836801443, 27.58676280]


def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    # if -2 < b < 2:
    #     return r'${}$'.format(a * 10 ** b)
    return r'${} · 10^{{{}}}$'.format(a, b)  # \times


# 雨滴谱反演色卡
colormap = {
    'dm': {
        "levels": (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5),
        "colors": ("#00ecec", "#01a0f6", "#0000f6", "#00ff00", "#3bc83b", "#009000", "#ffff00",
                   "#e7c000", "#ff9000", "#ff6633", "#ff5757", "#ff0000", "#c00000"),
        "label": "mm",
        "unit": "mm",
        "name": "粒子大小"
    },
    'nw': {
        "levels": (0, 1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000),
        "colors": ("#00ecec", "#01a0f6", "#0000f6", "#00ff00", "#3bc83b", "#009000", "#ffff00",
                   "#e7c000", "#ff9000", "#ff6633", "#ff5757", "#ff0000", "#c00000"),
        "label": "mm${^-}$${^1}$m${^-}$${^3}$",
        "unit": "mm﹣¹·m﹣³",
        "name": "粒子数密度"
    },
    'qpr': {
        "levels": (-1, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0,
                   40.0, 45.0, 50.0, 60.0, 80.0, 100.0),
        "colors": ("#ffffff", "#a6caf0", "#60c0c0", "#00c0c0", "#20a0c0", "#4060c0", "#204080", "#0000c0",
                   "#008000", "#40a000", "#20c000", "#40e040", "#c0c000", "#e0e000", "#ffff00", "#e0a0c0",
                   "#e08080", "#e06040", "#ff0000", "#ff00ff", "#800080"),
        "label": "mm/h",
        "unit": "mm/h",
        "name": "降水强度"
    }
}


class DsdRainRateController:
    def __init__(self, fileinfo_list, element, level, jobid, threshold=None):
        self.fileinfo_list = fileinfo_list
        self.element = element
        self.level = level
        self.jobid = jobid
        self.height_list = None  # 存储高度信息
        self.threshold = threshold  # 阈值
        path_cfg = config_info.get_dsd_rainrate_config
        base_path = os.path.dirname(os.path.dirname(__file__))
        self.exe_path = os.path.join(base_path, 'exe', 'estimate_dsd_rainrate_alldata_',
                                     'estimate_dsd_rainrate_alldata_', 'estimate_DSD_RainRate_alldata.exe')
        # path_cfg["exe_path"]
        self.fusion_exe_path = os.path.join(base_path, 'exe', 'eatimate_dsd_dainrate_fusion', 'HC_grid.exe')  # path_cfg["fusion_exe_path"]
        self.output_data_path = path_cfg['exe_out_path']  # 文件输出路径
        self.pic_file_path = os.path.join(path_cfg["pic_path"], str(jobid))
        self.input_dir = path_cfg['input_dir']
        self.temp_path = os.path.join(self.output_data_path, '_temp')
        if not Path(self.temp_path).exists():
            Path.mkdir(Path(self.temp_path), parents=True, exist_ok=True)

    def run_exe(self, fileinfos, time_str):
        """
        运行外部程序，生成结果文件, 返回生成文件路径
        """
        out_pol_dir = os.path.join(self.output_data_path, 'pol',
                                   time_str[0:4], time_str[4:6], time_str[6:8], time_str[8:10], time_str[10:12])
        if not Path(out_pol_dir).exists():
            Path.mkdir(Path(out_pol_dir), parents=True, exist_ok=True)
        out_dir = os.path.join(self.output_data_path, 'grb',
                               time_str[0:4], time_str[4:6], time_str[6:8], time_str[8:10], time_str[10:12])
        if not Path(out_dir).exists():
            Path.mkdir(Path(out_dir), parents=True, exist_ok=True)
        out_file = glob.glob(f"{out_dir}/Z*")
        out_inline_path = transfer_path(out_dir, is_win_path=True, is_inline_path=True)  # 远程挂载路径
        output_data_inline_files = glob.glob(f"{out_inline_path}/Z*")
        if output_data_inline_files:
            logger.info("文件{}无需执行算法程序".format(output_data_inline_files[-1]))
            return output_data_inline_files
        elif out_file:
            logger.info("文件{}无需执行算法程序".format(out_file[-1]))
            return out_file
        else:
            zwn = []
            fileids = []
            for fileinfo in fileinfos:
                zwn.append(fileinfo.get('equipNum'))
                fileids.append(fileinfo.get('fileId'))
                cmd = f"{self.exe_path} {fileinfo.get('filePath')} {out_pol_dir}"
                print(cmd)
                os.system(cmd)
                if not os.listdir(out_pol_dir):
                    raise ValueError(f"dsd_rainrate执行{out_pol_dir}失败")
                else:
                    out_pol_file = glob.glob(f"{out_pol_dir}/Z*")
                    temp_file_info = temp_message(out_pol_file, [fileinfo.get('fileId')])
                    publish_temp_msg("20", self.jobid, "DsdRainRate", temp_file_info)
            with Dataset(out_pol_file[0]) as ds:
                times = datetime.fromtimestamp(ds.RadTime).strftime('%Y%m%d%H%M%S')
            zwn.sort()
            zwn_s = ','.join(zwn)
            cmd = f"{self.fusion_exe_path} MS {times} {times} 60;60 {out_pol_dir}\\;{zwn_s}  {self.temp_path}\\ {out_dir}\\ HM_WN.ini 0;1 WEINI;250;-999;-999;-999;-999"
            print(cmd)
            os.system(cmd)
            if not os.listdir(out_dir):
                raise ValueError(f"dsd_rainrate执行{out_dir}失败")
            else:
                out_file = glob.glob(f"{out_dir}/Z*")
                temp_file_info = temp_message(out_file, fileids)
                publish_temp_msg("20", self.jobid, "DsdRainRate", temp_file_info)
            return out_file

    def draw(self, lon, lat, data, file_time):
        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8))

        lons, lats = np.meshgrid(lon, lat)
        my_map = Basemap(projection='merc', llcrnrlat=str_point[1], urcrnrlat=end_point[1],
                         llcrnrlon=str_point[0], urcrnrlon=end_point[0])
        lons, lats = my_map(lons, lats)

        cmap = ListedColormap(colormap[self.element]["colors"])
        norm = BoundaryNorm(boundaries=colormap[self.element]["levels"], ncolors=len(colormap[self.element]["levels"]))

        my_map.contourf(lons, lats, data, levels=colormap[self.element]["levels"],
                        colors=colormap[self.element]["colors"], extend='both')
        # ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading='auto')
        ax.axis('off')
        Path.mkdir(Path(self.pic_file_path), parents=True, exist_ok=True)
        png_path = os.path.join(self.pic_file_path, f"{file_time}_{self.element}.png")
        plt.savefig(png_path, dpi=400, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)

        args = [{"argsType": "height",
                 "defaultValue": float(self.height_list[4]),
                 "specialArgs": np.array(self.height_list).tolist(),
                 "unit": "m"}]
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": self.element, "elevation": float(self.level),
                    "otherData": {"args": None if self.element == "qpr" else args,
                                  "gisPicLonLat": [str_point, end_point]
                                  }}
        return pic_info

    def draw_point_time_height(self, file_time_list, height_list, data, eff_time=None, hail_time=None):
        # 绘制单点时间高度分析图
        data = np.array(data)
        data[data <= -999] = np.nan
        fig, ax = plt.subplots(figsize=(7, 5))
        level = colormap[self.element].get('levels')
        color = colormap[self.element].get('colors')
        title = colormap[self.element].get('name')
        x_label_list = [datetime.strptime(t[:], '%Y%m%d%H%M%S') for t in file_time_list]

        cmap = ListedColormap(color)
        norm = BoundaryNorm(boundaries=level, ncolors=len(level))
        im = ax.pcolormesh(x_label_list, np.array(height_list) / 1000, np.array(data).T, cmap=cmap, norm=norm, shading='nearest')
        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

        # x_label = [tt.strftime('%H:%M') for tt in x_label_list[::12]]
        # ax.set_xticks(ticks=np.arange(0, len(x_label_list))[::12])
        # ax.set_xticklabels(x_label)
        kwargs_major, kwargs_minor, timestyle = set_x_time(x_label_list)
        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)
        fig.autofmt_xdate(rotation=45)  # 自动调整x轴标签的角度

        title_time = '~'.join(
            [f'{x_label_list[0].strftime("%Y-%m-%d %H:%M")}', f'{x_label_list[-1].strftime("%Y-%m-%d %H:%M")}'])
        ax.set_title(f'{title}时间高度图  {title_time}', fontsize=FONTSIZE)
        # ax.set_xlabel(f'{x_label_list[0].strftime("%Y-%m-%d")}')
        ax.set_ylabel('高度 (km)')
        if self.element == 'nw':
            br = fig.colorbar(im, ticks=level, shrink=0.9, pad=0.05, format=mpl.ticker.FuncFormatter(fmt))
        else:
            br = fig.colorbar(im, ticks=level, shrink=0.9, pad=0.05)  # cax=position,
        br.ax.set_title(colormap[self.element].get('label'), position=(0.7, 0), fontsize=10)  # , position=(0.8, 0)

        png_path = os.path.join(self.pic_file_path, f"{file_time_list[0]}_{self.element}.png")
        if not os.path.isdir(self.pic_file_path):
            os.makedirs(self.pic_file_path)
        plt.savefig(png_path, dpi=300, transparent=False, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": f"{file_time_list[0]}_{self.element}.png",
                    "path": transfer_path(png_path, is_win_path=True),
                    "element": self.element, "elevation": self.level}
        return pic_info

    def draw_lines(self, file_time_list, y, eff_time=None, hail_time=None):
        """

        :param file_time_list: x 数组[]
        :param y: 数据数组
        :param eff_time: 作业时间 [[,], [,]]
        :param hail_time: 降雹时间 [[,], [,]]
        :return:
        """
        x_label_list = [datetime.strptime(t[:], '%Y%m%d%H%M%S') for t in file_time_list]
        y = np.array(y)
        title = colormap[self.element].get('name')
        fig, ax = plt.subplots(1, 1, sharey='all', figsize=(7, 2))
        plt.rcParams['font.size'] = 10
        ax.plot(x_label_list, y, marker=".")  # , markersize=2
        ax_add_time_range(ax, eff_time, alpha=0.3, color='red')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='purple')

        # 绘制主副坐标轴
        if ax.get_ylim()[-1] > 1000:
            ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        kwargs_major, kwargs_minor, timestyle = set_x_time(x_label_list)
        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)
        fig.autofmt_xdate(rotation=45)  # 自动调整x轴标签的角度

        title_time = '~'.join(
            [f'{x_label_list[0].strftime("%Y-%m-%d %H:%M")}', f'{x_label_list[-1].strftime("%Y-%m-%d %H:%M")}'])
        ax.set_title(f'{title}  {title_time}', fontsize=FONTSIZE)
        # ax.set_xlabel(f'{x_label_list[0].strftime("%Y-%m-%d")}')
        ax.set_ylabel(colormap[self.element].get('label'))

        png_path = os.path.join(self.pic_file_path, f"{file_time_list[0]}_{self.element}.png")
        if not os.path.isdir(self.pic_file_path):
            os.makedirs(self.pic_file_path)
        plt.savefig(png_path, dpi=300, transparent=False, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": f"{file_time_list[0]}_{self.element}.png",
                    "path": transfer_path(png_path, is_win_path=True),
                    "element": self.element, "elevation": self.level}
        return pic_info

    def get_data(self, nc_file):
        with nc.Dataset(nc_file, 'r') as ds:
            if not self.height_list:
                self.height_list = list(ds.variables['Dim1'][:])
            lat = ds.variables['Dim2'][:]
            lon = ds.variables['Dim3'][:]
            times = datetime.fromtimestamp(ds.RadTime).strftime('%Y%m%d%H%M%S')
            ele_data = ds.variables[self.element][:]
            ele_data[ele_data <= -999] = None
            if self.threshold is not None:
                ele_data = select_data(ele_data, self.element, self.threshold)
        return ele_data, lon, lat, times

    def run(self, point_lat=None, point_lon=None, eff_time=None, hail_time=None):
        # 执行程序
        pic_file = []
        file_time_list = []
        data = []
        data_file = []
        nc_file = []
        for files in self.fileinfo_list:
            for file in files:
                file_id = file.get('fileId')
                if file.get('radType') == 'HTM':
                    time_str = re.search(r'\d{8}_\d{2}', os.path.basename(file.get('filePath'))).group().replace('_',
                                                                                                                 '')
                    time_str = (datetime.strptime(time_str, '%Y%m%d%H') - timedelta(hours=8)).strftime('%Y%m%d')
                else:
                    time_str = re.search(r'\d{8}', os.path.basename(file.get('filePath'))).group()  # 源文件为世界时
                # input_dir = os.path.join(self.input_dir, time_str, str(file_id))
                # nc_file = os.path.join(input_dir, os.listdir(input_dir)[0])
                # 本地若不存在则查询是否上传到minio，文件上传可能存在延迟
                t_str = re.search(r'\d{14}', os.path.basename(file.get('filePath'))).group()
                t_str = (datetime.strptime(t_str, '%Y%m%d%H%M%S') - timedelta(minutes=1)).strftime('%Y%m%d%H%M%S')
                temp_path = os.path.join(self.input_dir, time_str, str(file_id),
                                         f'Z_RADR_I_{file.get("equipNum")}_{t_str}_O_DOR_ETW_CAP_FMT.nc')
                try:
                    nc_file = file_exit(transfer_path(temp_path, is_win_path=True), sleep_times=2)
                except NameError:
                    nc_file = file_exit(transfer_path(temp_path, is_win_path=True).replace("CAP", "CRA"), sleep_times=2)
                file['filePath'] = nc_file

            data_file_temp = self.run_exe(files, t_str)
            data_file.extend(data_file_temp)
            if data_file:
                try:
                    ele_data, lons, lats, file_time = self.get_data(data_file_temp[0])  # 获取要素数据
                except RuntimeError:
                    os.remove(data_file_temp[0])
                    raise FileNotFoundError("文件错误，请重新生成。")
            else:
                raise Exception("程序执行失败，未生成文件")

            # 如果传入点经纬度参数，则取该点的历史数据返回
            if point_lat and point_lon:
                point_data = get_point_data(lats, lons, ele_data, point_lat, point_lon)
                file_time_list.append(file_time)
                if self.element == "qpr":
                    point_data = point_data[0]  # 取雨强单层数据

                data.append(np.array(point_data).tolist())
            else:
                try:
                    if self.level is None:
                        self.level = self.height_list[4]
                    idx = 0 if self.element == "qpr" else np.argmin(abs(np.array(self.height_list) - self.level))
                except IndexError:
                    raise Exception("高度参数错误")
                pic_file.append(self.draw(lons, lats, ele_data[idx], file_time))

        # 如果需要绘制时间高度图，需要对点数据进行出图
        if point_lat and point_lon:
            if self.element == "qpr":
                y = np.around(data, 2).tolist()
                pic_data = [
                    {"element": self.element, "x": file_time_list, "y": [y], "xlabel": "",
                     "ylabel": colormap[self.element].get("unit"), "yname": [""]}]
                pic_file.append(self.draw_lines(file_time_list, y, eff_time=eff_time, hail_time=hail_time))
                # pic_file.append(self.draw_point_time_height(file_time_list, self.height_list, data))
                return [{"picFiles": pic_file, "picData": pic_data}]
            if self.level:
                idx = np.argmin(abs(np.array(self.height_list) - self.level))
                ele_data = [i_data[idx] for i_data in data]
                pic_file.append(self.draw_lines(file_time_list, ele_data, eff_time=eff_time, hail_time=hail_time))
                pic_data = [
                    {"element": self.element, "x": file_time_list, "y": [np.around(ele_data, 2).tolist()], "xlabel": "",
                     "ylabel": colormap[self.element].get("unit"), "yname": [""]}]
                return [{"picFiles": pic_file, "picData": pic_data}]
            # 其他要素需要绘制时间高度图
            pic_file.append(self.draw_point_time_height(file_time_list, self.height_list, data,
                                                        eff_time=eff_time, hail_time=hail_time))

        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, data_file)

        return [{"picFiles": pic_file}]
