# -*- coding: utf-8 -*-
import os
import re
import shutil
import time

import matplotlib as mpl
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib import font_manager
from matplotlib.dates import DateFormatter, MinuteLocator
import zipfile
import sys

from config.config import config_info, cr_line_config, cr_pic_config, font_path
from mq.pubulisher import publish_temp_msg
from service.dsd_rainrate import fmt
from service.utils import transfer_path, temp_message, ax_add_time_range

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.size'] = 12
mpl.use('Agg')
FONTSIZE = 8
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


def search_index(item_list, tar, left, right):
    """
    在有序列表中找到目标值在数据列表中距离最近的索引
    :param item_list: 数据列表
    :param tar: 目标值
    :param left: 数据列表的左边界
    :param right: 数据列表的右边界
    :return: 目标值在数据列表中距离最近的索引
    """
    # 如果目标值不在有序列表范围内，则返回None
    if tar > item_list[-1] or tar < item_list[0]:
        return None
    index = (left + right) // 2
    if item_list[index] <= tar <= item_list[index + 1]:
        if abs(tar - item_list[index]) < abs(tar - item_list[index + 1]):
            return index
        else:
            return index + 1
    elif item_list[index] > tar:
        return search_index(item_list, tar, left, index - 1)
    else:
        return search_index(item_list, tar, index + 1, right)


def unzip_file(file, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    f = zipfile.ZipFile(file, 'r')  # 压缩文件位置
    real_file = f.extract(f.namelist()[0], out_path)  # 解压位置
    f.close()
    return real_file


def draw(lons, lats, data, pic_file, levels, colors, units, title, data_type="prod", pic_type="VOL", polar_line=False,
         ylims=None, eff_time=None, hail_time=None):
    """
    绘制雷达色斑图
    :param lons: 经度数组(lon,lat)
    :param lats: 纬度数组(lon,lat)
    :param data: 数据数组(lon,lat)
    :param pic_file: 要保存的图片本地地址
    :param levels: 色值levels
    :param colors: 色值levels对应的色值colors
    :param units: 单位
    :param title: 标题
    :param data_type: 数据类型（grid/prod）
    :param pic_type: 绘图类型(VOL,RHI,THI)
    :param polar_line: 是否画极坐标线
    :param ylims: 纵轴范围
    :return:
    """
    if ylims is None:
        ylims = [np.nanmin(lats), np.nanmax(lats)]
    fig, ax = plt.subplots(figsize=(8, 3))
    pad_inches = 0.2
    transparent = False

    if data_type == "prod":
        # 补全径向360-0的色块
        lons = np.row_stack((lons, lons[0]))
        lats = np.row_stack((lats, lats[0]))
        data = np.row_stack((data, data[0]))

    if levels and colors:
        im = ax.contourf(lons, lats, data, levels=levels, colors=colors, extend='max')
        # cmap = ListedColormap(colors)
        # norm = BoundaryNorm(boundaries=levels,
        #                     ncolors=len(levels))
        # im = ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, shading='nearest')
    else:
        data[data == 0] = None
        im = ax.contourf(lons, lats, data, cmap='rainbow', extend='max')
    # ax.set_xticklabels(lons[0, ::6])
    # ============================添加网格以及色卡显示=================================
    if pic_type == "RHI":
        # ax.set(xlabel='距离(km)', ylabel='高度(km)')
        plt.title(title, fontsize=FONTSIZE)
        plt.xlabel('距离(km)', fontsize=10)
        plt.ylabel('高度(km)', fontsize=10)
        plt.ylim(ylims)
        # position = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # (左，下，宽，高)
        br = plt.colorbar(im, fraction=0.033, pad=0.02, aspect=30, ticks=levels)
        ax.tick_params(labelsize=8)
        br.ax.set_title(units, fontsize=12)
        br.ax.tick_params(labelsize=10)
    elif pic_type == "THI":
        # ax.set(xlabel=lons[0].strftime('%Y-%m-%d'), ylabel='高度(km)')
        # plt.xlabel(lons[0].strftime('%Y-%m-%d'), fontsize=10)
        plt.ylabel('高度(km)', fontsize=10)
        plt.ylim(ylims)
        if 'raw' in pic_file:
            title = title.replace('-', '\t') + "质控前"
        plt.title(f'{title}\t{lons[0].strftime("%Y-%m-%d %H:%M")}~{lons[-1].strftime("%Y-%m-%d %H:%M")}',
                  fontsize=FONTSIZE)
        # position = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # (左，下，宽，高)
        br = plt.colorbar(im, fraction=0.033, pad=0.02, aspect=30, ticks=levels)
        ax.tick_params(labelsize=8)
        br.ax.set_title(units, fontsize=12)
        br.ax.tick_params(labelsize=10)

        majorformatter = DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(majorformatter)
        ax.xaxis.set_minor_locator(MinuteLocator(interval=5))
        # ax.set_xlabel(lons[0].strftime('%Y-%m-%d'))
        ax.tick_params(axis="x", direction="out", which='major', length=8, width=1.5)
        ax.tick_params(axis="x", direction="out", which='minor', length=4, width=1.0)
        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')
        # ax.set_ylabel('高度 (km)')
    elif pic_type == 'VOL':
        # position = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # (左，下，宽，高)
        # br = fig.colorbar(im, cax=position, ticks=levels)
        # br.ax.set_title(units)
        pad_inches = 0
        transparent = True
        ax.axis('off')

    # ============================添加极坐标线=================================
    if polar_line:
        ax.axis('off')
        ax1 = fig.add_axes([0.13, 0.111, 0.767, 0.767], polar=True)
        # ax1.set_theta_zero_location('N')  # 使用set_theta_zero_location()使小节从北方开始
        # ax1.set_theta_direction(-1)  # 使用set_theta_direction()使条形顺时针旋转.
        # ax1.set_rlabel_position(0)  # 使用set_rlabel_position()移动径向标签.
        angle_num = 4  # 将圆36等份
        ax1.set_thetagrids(range(0, 360, int(360 / angle_num)), labels=[""] * angle_num)  # 设置角度刻度标签（这里角度标签labels都为空）
        # ax1.set_thetagrids(range(0, 360, int(360 / angle_num)))  # 设置角度刻度标签（这里角度标签labels都为空）
        distance_list = [2, 5, 10]  # 距离圈列表，单位km
        # ax1.set_rgrids(distance_list, labels=[''] * len(distance_list))  # 设置距离标签（这里刻度标签labels都为空）
        ax1.set_rgrids(distance_list, labels=["{}km".format(i) for i in distance_list])  # 设置距离标签（这里刻度标签labels都为空）
    # =======================================================================================

    plt.savefig(pic_file, dpi=300, transparent=transparent, bbox_inches="tight", pad_inches=pad_inches)
    plt.close()
    # 上传图片
    # # minio_client.put_object(pic_file)


class CR(object):
    """读取云雷达算法程序生成的nc文件，绘制产品图"""

    def __init__(self, nc_file, f_type, title):
        self.fb = nc.Dataset(nc_file, "r")
        self.f_type = f_type
        self.title = title
        self.scan_type = self.fb.RadScan
        self.data_time = datetime.strptime(os.path.basename(nc_file)[:14], "%Y%m%d%H%M%S")
        # self.height_list = list(range(100, int(self.fb.ResolutionOfGridInZ * 1000) * (self.fb.GradNumNz + 1), 100))

    def get_data(self, element, elevation=None, height=None):
        """
        获取指定的数据及对应经纬度
        :param element: 要素名
        :param elevation: 仰角
        :param height: 高度层
        :return:
        """
        if self.f_type == "Grid":
            if self.scan_type == "VOL":
                height_list = list(range(100, int(self.fb.ResolutionOfGridInZ * 1000) * (self.fb.GradNumNz + 1), 100))
                height_index = height_list.index(height)
                lons = self.fb.variables["Lon_grid_data"][:]
                lats = self.fb.variables["Lat_grid_data"][:]
                datas = self.fb.variables[element][height_index]
                datas[datas == -999] = np.nan
            elif self.scan_type == "RHI":
                datas = self.fb.variables[element][:]
                lat, lon = datas.shape
                lons = np.arange(-int(lon / 2) * 0.1, (int(lon / 2) + 1) * 0.1, 0.1)
                lats = np.arange(0, lat * 0.1, 0.1)

                datas[datas == -999] = np.nan
                # datas = datas.T
            else:
                raise ValueError("请输入正确雷达类型")
            return lons, lats, datas
        else:
            # [5.0 8.0 16.7 30.0 80.0]
            if self.scan_type == "VOL":
                elevation_angle = list(self.fb.variables["Elevation_Angle"][:].astype(str)).index(str(elevation))
                lon = self.fb.variables["Lon_raw_data"][elevation_angle, :, :]
                lat = self.fb.variables["Lat_raw_data"][elevation_angle, :, :]
                data = self.fb.variables[element][elevation_angle, :, :]
            elif self.scan_type == "RHI":

                elevation_angle = self.fb.variables["Elevation_Angle"][:]
                lon = self.fb.variables["Lon_raw_data"][0, :, :]
                lat = self.fb.variables["Lat_raw_data"][0, :, :]
                data = self.fb.variables[element][0, :, :]
                data = data.T
            elif self.scan_type == "THI":
                elevation_angle = []
                str_s = self.fb.getncattr('Starting_Time(Seconds)')
                end_s = self.fb.getncattr('End_Time(Seconds)')
                str_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(str_s))
                end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_s))
                try:
                    time_s = int((end_s - str_s) / (len(self.fb.variables['Azimuth_Angle']) - 1))
                except KeyError:
                    time_s = 1
                lon = pd.date_range(start=str_time, end=end_time, freq=f'{time_s}s')
                lat = np.arange(0, self.fb.Bin_num * self.fb.getncattr('Distance_resolution(m)'),
                                self.fb.getncattr('Distance_resolution(m)'))
                try:
                    data = self.fb.variables[element + "1"][0, :, :]
                except KeyError:
                    data = self.fb.variables[element][0, :, :]
                data = data.T
                lon = lon[0: data.shape[1]]
            elif self.scan_type == "FFT":
                lon, lat, elevation_angle = [], [], []
                str_s = self.fb.getncattr('Starting_Time(Seconds)')
                end_s = self.fb.getncattr('End_Time(Seconds)')
                str_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(str_s))
                end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_s))
                try:
                    data = self.fb.variables[element + "1"][0, :, :]
                except KeyError:
                    data = self.fb.variables[element][0, :, :]
            else:
                raise ValueError("数据类型错误")

            data[data == -999] = np.nan
            return lon, lat / 1000, data, elevation_angle

    def get_pic(self, element, levels, colors, units, out_path, elevation=None, height=None, ylims=None,
                eff_time=None, hail_time=None):
        if self.f_type == "Prod":  # 极坐标
            lon, lat, data, elevation = self.get_data(element, elevation)
            pic_file = "{}_{}_prod_{}.png".format(element, self.data_time.strftime("%Y%m%d%H%M%S"), self.scan_type)
            polar_line = True
            if self.scan_type == "RHI":
                bin_num = self.fb.dimensions['Binnum'].size
                dim3 = np.arange(0, bin_num * 30, 30)
                deg2rad = np.pi / 180
                lon = dim3.reshape(len(dim3), 1) * np.cos(elevation.reshape(1, len(elevation)) * deg2rad) / 1000
                lat = dim3.reshape(len(dim3), 1) * np.sin(elevation.reshape(1, len(elevation)) * deg2rad) / 1000
                polar_line = False
            if self.scan_type == "THI":
                polar_line = False
        else:  # 格点Grid
            lon, lat, data = self.get_data(element, height=height)
            pic_file = "{}_{}_grid_{}.png".format(element, self.data_time.strftime("%Y%m%d%H%M%S"), self.scan_type)
            polar_line = False
        if os.path.isfile(pic_file):
            logger.info("图片{}已存在，无需重复生成".format(pic_file))
            return pic_file
        pic_file = os.path.join(out_path, pic_file)
        draw(lon, lat, data, pic_file, levels, colors, units, self.title, data_type=self.f_type,
             pic_type=self.scan_type, polar_line=polar_line, ylims=ylims, eff_time=eff_time, hail_time=hail_time)
        logger.info(f"生成图片：{pic_file}")
        return pic_file, data

    # def __del__(self):
    #     self.fb.close()


class CRController(object):
    def __init__(self, file_list, jobid, element, title, scan_t, height=None, elevation=None,
                 point_lat=None, point_lon=None, analysis_type=None, ylims=None):
        self.new_element = ("IWCgrid", "LWCgrid", "RWCgrid", "SWCgrid", "refgrid", "velgrid", "swgrid")
        self.element_dic = {"vel": "径向速度", "ref": "回波强度", "sw": "速度谱宽", "IWC": "冰云含量",
                            "LWC": "液体水含量", "RWC": "雨含水量",
                            "SWC": "雪含水量"}
        # self.ori_file_list = [file[0].get('filePath') for file in file_list]
        # self.ori_file_list.sort()
        self.ori_file_list = file_list
        self.jobid = jobid
        self.element = element
        self.title = title
        self.height = height
        self.elevation = elevation
        self.point_lat = point_lat
        self.point_lon = point_lon
        type_c = ['VOL', 'SAN', 'RHI', 'RTI', 'FFT']
        if scan_t:
            scan_type = type_c[scan_t - 1]
        else:
            scan_type = type_c[0]
        self.scan_type = scan_type
        if self.scan_type == 'RTI':
            self.analysis_type = 3
            # self.element = elemen
        else:
            self.analysis_type = analysis_type
        self.ylims = ylims
        self.prod_files = []
        self.grid_files = []

        path_cfg = config_info.get_cr_config
        self.output_data_path = path_cfg["exe_out_path"]  # 输出路径
        self.pic_file_path = path_cfg["pic_path"]  # 生成图片保存的路径
        self.exe_path = path_cfg["exe_path"]  # os.path.join(BASE_PATH, 'exe', 'CR', 'CR2.exe')  exe路径
        self.minio_disk = path_cfg["minio_disk"]  # mini路径需要替换为本地盘符
        self.tmp_path = os.path.join(path_cfg["tmp_path"], str(self.jobid))  # 临时文件路径，存放原始数据解压后的文件

    def exe(self):

        for file_info in self.ori_file_list:
            input_file_id = file_info[0].get('fileId')
            f = file_info[0].get('filePath')
            # zip_file = f.replace(self.minio_disk, self.pic_file_path[:2])

            output_path = os.path.join(self.output_data_path, str(file_info[0].get('fileId')))
            data_time = re.search(r'\d+', os.path.basename(f).split(".")[0].replace("_", "")).group()[-14:]
            out_file_prod = os.path.join(output_path, "{}Cr{}Prod.NC".format(data_time, self.scan_type))
            out_file_grid = os.path.join(output_path, "{}Cr{}Grid.NC".format(data_time, self.scan_type))
            if os.path.isfile(transfer_path(out_file_prod, is_win_path=True, is_inline_path=True)) or \
                    os.path.isfile(transfer_path(out_file_grid, is_win_path=True, is_inline_path=True)):
                logger.info("文件{}无需执行算法程序".format(f))
                self.prod_files.append(transfer_path(out_file_prod, is_win_path=True, is_inline_path=True))
                self.grid_files.append(transfer_path(out_file_grid, is_win_path=True, is_inline_path=True))
                continue
            output_path = os.path.join(output_path, str(self.jobid))
            out_file_prod = os.path.join(output_path, "{}Cr{}Prod.NC".format(data_time, self.scan_type))
            out_file_grid = os.path.join(output_path, "{}Cr{}Grid.NC".format(data_time, self.scan_type))
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            zip_file = transfer_path(f)
            # zip_file = file_exit(f)  # 本地若不存在则查询是否上传到minio，文件上传可能存在延迟
            # 解压缩
            try:
                input_file = unzip_file(zip_file, self.tmp_path)
            except zipfile.BadZipFile:
                input_file = zip_file
            if os.path.getsize(input_file) == 0:
                raise FileExistsError("原始数据异常")
            # cmd = "{} && cd {} && {} {} {} {}".format(
            #     os.path.dirname(self.exe_path)[:2], os.path.dirname(self.exe_path),
            #     os.path.basename(self.exe_path), input_file, output_path, SCA_DAT)
            SCA_DAT = os.path.join(os.path.dirname(self.exe_path), 'SCA-ku_ka_w.DAT')
            cmd = f'{self.exe_path} {input_file} {output_path} {SCA_DAT}'
            print(cmd)
            os.system(cmd)
            if os.path.isfile(out_file_prod) or os.path.isfile(out_file_grid):
                self.prod_files.append(out_file_prod)
                self.grid_files.append(out_file_grid)
                #  ---中间文件生成消息发送---
                if self.scan_type in ['RTI', 'FFT']:
                    temp_file_info = temp_message(out_file_prod, input_file_id, jobid=self.jobid)
                else:
                    temp_file_info = temp_message([out_file_grid, out_file_prod], input_file_id, jobid=self.jobid)
                publish_temp_msg("20", self.jobid, "CR", temp_file_info)

            else:
                logger.info("算法程序执行失败，未生成数据:{}、{}".format(out_file_prod, out_file_grid))
                raise Exception("源文件错误，未生成数据")

    def get_prod_pic(self, eff_time=None, hail_time=None):
        ele = re.sub('(grid|qc|raw)', '', self.element)
        levels = cr_pic_config[ele]["levels"]
        colors = cr_pic_config[ele]["colors"]
        units = cr_pic_config[ele]["unit"]
        pic_list = []
        # 极坐标质控前后的产品图（ref:回波强度、vel:径向速度、sw:速度谱宽）
        for f in self.prod_files:
            f_type = os.path.basename(f).split(".")[0][-4:]  # 数据类型（prod：极坐标数据，Grid：网格数据）
            cr = CR(f, f_type, self.title)
            # 绘制质控前产品图
            # np.array(cr.fb.variables['Azimuth_Angle']).tolist()
            # 雷达位置信息
            if self.scan_type == 'VOL':
                rad_height = float(cr.fb.Radar_Station_height)
                rad_lat = float(cr.fb.Radar_Lat)
                rad_lon = float(cr.fb.Radar_Lon)
                elevation_list = [float(str(e)) for e in cr.fb.variables['Elevation_Angle']]  # 仰角信息
                bin_num = cr.fb.dimensions['Binnum'].size  # 径向距离库数
                Gatewidth = 30  # 距离库长

                otherdata = {"args": [{
                    "argsType": "height", "defaultValue": 5.0,
                    "specialArgs": [5.0, 8.0, 16.7, 30.0, 80.0], "unit": "°"}],
                    "polarPicLonLat": {"lon": 104.09,
                                       "lat": 27.05,
                                       "radius": 10000}
                }
            else:
                otherdata = {"args": None}

            if self.element.endswith('raw'):
                pic_path = os.path.join(self.pic_file_path, str(self.jobid), "old")
                if not os.path.isdir(pic_path):
                    os.makedirs(pic_path)
                pic, data = cr.get_pic(self.element, levels, colors, units, pic_path, elevation=self.elevation,
                                       height=self.height, ylims=self.ylims, eff_time=eff_time, hail_time=hail_time)
                if np.isnan(np.nanmax(data)):
                    max_data = 0
                else:
                    max_data = float(np.nanmax(data))
                otherdata["maxThreshold"] = max_data
                pic_list.append({
                    "element": self.element, "fileName": os.path.basename(pic),
                    "path": transfer_path(pic, is_win_path=True),
                    "img_type": "figure", "otherData": otherdata
                    # "args": [{"argsType": "elevation", "defaultValue": self.elevation,
                    #           "specialArgs": [s], "unit": "°"}]
                })
            # 绘制质控后产品图
            elif self.element.endswith('qc'):
                pic_path2 = os.path.join(self.pic_file_path, str(self.jobid), "new")
                if not os.path.isdir(pic_path2):
                    os.makedirs(pic_path2)
                pic, data = cr.get_pic(self.element, levels, colors, units, pic_path2, elevation=self.elevation,
                                       height=self.height, ylims=self.ylims, eff_time=eff_time, hail_time=hail_time)
                if np.isnan(np.nanmax(data)):
                    max_data = 0
                else:
                    max_data = float(np.nanmax(data))
                otherdata["maxThreshold"] = max_data
                pic_list.append({
                    "element": self.element, "fileName": os.path.basename(pic),
                    "path": transfer_path(pic, is_win_path=True),
                    "img_type": "figure", "otherData": otherdata
                })
            else:
                pic_path3 = os.path.join(self.pic_file_path, str(self.jobid), "thi")
                if not os.path.isdir(pic_path3):
                    os.makedirs(pic_path3)
                pic, data = cr.get_pic(self.element, levels, colors, units, pic_path3, elevation=self.elevation,
                                       height=self.height, ylims=self.ylims, eff_time=eff_time, hail_time=hail_time)
                if np.isnan(np.nanmax(data)):
                    max_data = 0
                else:
                    max_data = float(np.nanmax(data))
                pic_list.append({
                    "element": self.element, "fileName": os.path.basename(pic),
                    "path": transfer_path(pic, is_win_path=True),
                    "img_type": "figure", "otherData": {"maxThreshold": max_data}
                })
            cr.fb.close()
        return pic_list

    def get_grid_pic(self):
        ele = re.sub('(grid|qc|raw)', '', self.element)
        levels = cr_pic_config[ele]["levels"]
        colors = cr_pic_config[ele]["colors"]
        units = cr_pic_config[ele]["unit"]
        pic_list = []
        # 极坐标质控前后的产品图（ref:回波强度、vel:径向速度、sw:速度谱宽）
        for f in self.grid_files:
            f_type = os.path.basename(f).split('.')[0][-4:]  # 数据类型（prod：极坐标数据，Grid：网格数据）
            # 绘制质控前产品图
            pic_path = os.path.join(self.pic_file_path, str(self.jobid))
            if not os.path.isdir(pic_path):
                os.makedirs(pic_path)
            cr = CR(f, f_type, self.title)
            pic, data = cr.get_pic(self.element, levels, colors, units, pic_path, height=self.height, ylims=self.ylims)
            if np.isnan(np.nanmax(data)):
                max_data = 0
            else:
                max_data = float(np.nanmax(data))
            # cr = CR(f, f_type, self.title)
            # pic = cr.get_pic(self.element, levels, colors, units, pic_path, height=self.height)
            rad_height = float(cr.fb.Radar_Station_height)
            rad_lat = float(cr.fb.Radar_Lat)
            rad_lon = float(cr.fb.Radar_Lon)
            cr.fb.close()

            s_point = [103.01042080, 25.960774]
            d_point = [104.836801443, 27.58676280]
            otherdata = {"args": [{
                "argsType": "height", "defaultValue": 500,
                "specialArgs": np.arange(100, 1700, 100).tolist(), "unit": "m"}],
                "gisPicLonLat": [s_point, d_point],
                "maxThreshold": max_data
            }
            pic_list.append({
                "element": self.element, "fileName": os.path.basename(pic),
                "path": transfer_path(pic, is_win_path=True),
                "img_type": "figure", "otherData": otherdata
            })
        return pic_list

    def get_point_data(self):
        data_time_list = []
        point_data_list = []
        height_list = None
        for f in self.grid_files:
            f_type = os.path.basename(f).split('.')[0][-4:]  # 数据类型（prod：极坐标数据，Grid：网格数据）
            cr = CR(f, f_type, self.title)
            data_time_list.append(cr.data_time.strftime("%Y%m%d%H%M%S"))
            height_list = list(range(100, int(cr.fb.ResolutionOfGridInZ * 1000) * (cr.fb.GradNumNz + 1), 100))
            height_list = [he / 1000 for he in height_list]  # 转化为千米
            lat = cr.fb.variables["Lat_grid_data"]
            lon = cr.fb.variables["Lon_grid_data"]
            data = cr.fb.variables[self.element + "grid"]
            cr.fb.close()
            # 获取目标点的数据
            lat_list = lat[:, 0]
            lon_list = lon[0]
            lat_index = search_index(lat_list, self.point_lat, 0, len(lat_list) - 1)
            lon_index = search_index(lon_list, self.point_lon, 0, len(lon_list) - 1)
            if not lat_index or not lon_index:
                raise Exception("该点（{},{}）不在目标范围内".format(np.round(self.point_lat, 2), np.round(self.point_lon, 2)))
            point_data_list.append(data[:, lat_index, lon_index])
        return data_time_list, point_data_list, height_list

    def draw_point_time_height(self, file_time_list, height_list, data, title, unit):
        # 绘制单点时间高度分析图
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = ['DengXian']
        plt.rcParams['font.sans-serif'] = prop.get_name()  # 使图正常显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 使刻度正常显示正负号

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(title)
        ax.set_xlabel("时间")
        ax.set_ylabel("高度(km)", loc='center')

        data[data == -999] = np.nan
        # level = colormap[self.element].get('levels')
        # cmap = ListedColormap(colormap[self.element].get('colors'))
        # norm = BoundaryNorm(boundaries=level, ncolors=len(level))
        # im = ax.pcolormesh(file_time_list, height_list, data, cmap=cmap, norm=norm, shading='gouraud')
        im = ax.pcolormesh(file_time_list, height_list, data)
        br = fig.colorbar(im, ax=ax)
        br.ax.set_title(unit)
        fig.autofmt_xdate(rotation=45)

        png_path = os.path.join(self.pic_file_path, f"{file_time_list[0]}_{self.element}.png")
        plt.savefig(png_path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # # minio_client.put_object(png_path)
        pic_info = {
            "element": self.element, "fileName": os.path.basename(png_path),
            "path": transfer_path(png_path, is_win_path=True),
            "img_type": "figure"
        }
        return pic_info

    def draw_fft(self, element):
        if element.endswith('raw'):
            pic_path = os.path.join(self.pic_file_path, str(self.jobid), "old")
            title = '质控前'
        else:
            pic_path = os.path.join(self.pic_file_path, str(self.jobid), "new")
            title = '质控后'
        if not os.path.isdir(pic_path):
            os.makedirs(pic_path)

        for f in self.prod_files:
            f_type = os.path.basename(f).split(".")[0][-4:]  # 数据类型（prod：极坐标数据，Grid：网格数据）
            cr = CR(f, f_type, self.title)
            pic_file = "{}_{}_grid_{}.png".format(element, cr.data_time.strftime("%Y%m%d%H%M%S"), cr.scan_type)
            _, _, data, elevation = cr.get_data(element)
            # data = ds.variables['SZQC1'][0]
            # data = ds.variables['SZraw1'][0]
            data[data == 0] = None
            data = data.T
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.rcParams['font.size'] = 15
            y = np.linspace(0.030, data.shape[0] * 0.030, data.shape[0])
            if element.endswith('raw'):
                x = np.linspace(-11.2, 11.2, data.shape[1])
                data[data <= -90] = None
                im = ax.pcolormesh(x, y, data, cmap='rainbow')
                cb = fig.colorbar(im, fraction=0.03, pad=0.02, aspect=30)
            else:
                x = np.linspace(-11.2, 3 * 11.2, data.shape[1])
                im = ax.pcolormesh(x, y, data, norm=LogNorm(), cmap='rainbow')
                cb = fig.colorbar(im, fraction=0.03, pad=0.02, aspect=30)  # , format=mpl.ticker.FuncFormatter(fmt)

            # ax.set(xlabel='径向速度(m/s)', ylabel='高度(km)')
            plt.xlabel('径向速度(m/s)', fontsize=15)
            plt.ylabel('高度(km)', fontsize=15)
            plt.title(f'{title}功率谱\t{cr.data_time.strftime("%Y-%m-%d %H:%M:%S")}')
            pic_file = os.path.join(pic_path, pic_file)
            plt.savefig(pic_file, dpi=300, transparent=False, bbox_inches="tight", pad_inches=0.02)
            plt.close()
            cr.fb.close()
            # 上传图片
            # # minio_client.put_object(pic_file)

            return pic_file

    def run(self, eff_time=None, hail_time=None):
        logger.info("开始执行算法程序")
        # 执行算法程序
        self.exe()
        logger.info("算法程序执行完成")
        line_data = []
        pic_list = []

        if self.scan_type in ["FFT"]:
            plt.rcParams['font.size'] = 12
            for e in ["raw", "QC"]:
                pic = self.draw_fft(self.element.upper() + e)
                pic_list.append({
                    "element": self.element, "fileName": os.path.basename(pic),
                    "path": transfer_path(pic, is_win_path=True),
                    "img_type": "figure"
                })
        elif self.scan_type in ["RHI"]:
            pic_list = self.get_prod_pic()

        if self.analysis_type == 1:
            if not self.point_lat or not self.point_lon:
                raise Exception("廓线图绘制请输入坐标点")
            # 获取廓线图数据
            data_time_list, point_data_list, height_list = self.get_point_data()
            for i, data_time in enumerate(data_time_list):
                cr_line_config[self.element]["y"] = [height_list]
                x_data = np.array(point_data_list[i])
                x_data[x_data == -999] = np.nan
                cr_line_config[self.element]["x"] = x_data.tolist()
                cr_line_config[self.element]["yname"] = [self.title]
                line_data.append([cr_line_config[self.element]])

        elif self.analysis_type == 2:
            if not (self.element in self.new_element):
                raise Exception(f"格点变量参数为：{self.new_element}")
            logger.info("开始绘制网格产品图")
            pic_list = self.get_grid_pic()
            logger.info("网格产品图绘制完成")
        elif self.analysis_type == 3:
            logger.info("开始绘制极坐标产品图")
            pic_list = self.get_prod_pic(eff_time=eff_time, hail_time=hail_time)
            if self.element == 'refqc':
                self.element = "refraw"
                self.get_prod_pic()
            logger.info("极坐标产品图绘制完成")
        # if self.element in ("ref", "vel", "sw"):
        #     pic_list += self.get_prod_pic()
        # 单点分析
        # elif self.analysis_type == 4:
        #     if not self.point_lat or not self.point_lon:
        #         raise Exception("时间高度图绘制请输入坐标点")

        # 绘制时间高度图
        # data_time_list, point_data_list, height_list = self.get_point_data()
        # unit = cr_pic_config[self.element]["unit"]
        # pic_list.append(self.draw_point_time_height(data_time_list, height_list, np.array(point_data_list).T,
        #                                             self.title, unit))
        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, [self.prod_files, self.grid_files], self.jobid)
        try:
            shutil.rmtree(self.tmp_path)
        except FileNotFoundError:
            print("已删除")
        return [{"picFiles": pic_list, "picData": line_data}]

# if __name__ == '__main__':
# data_prod = "/Users/yangzhi/code/GZFB/code/radar_analysis_pic/out/CR/data/2/20200506001721radar_prod.NC"
# data_grid = "/Users/yangzhi/code/GZFB/code/radar_analysis_pic/out/CR/data/2/20200506001721radar_Grid.NC"
# cr = CR(data_grid, "grid")
# cr.get_pic("refgrid", height=2)
# cr = CR(data_prod, "prod")
# cr.get_pic("refraw", elevation=0.5)
# cr.get_pic("refqc", elevation=0.5)
# exit()

# input_filelist = [
#     {
#         "filePath": r"gzfb/radar_analysis_pic/data/CR/20200506/20200506_000245.00.002.001_R0.zip",
#         "jobFileId": 1,
#         "storeSource": "minio",
#         "fileId": 3,
#     },
#     {
#         "filePath": r"gzfb/radar_analysis_pic/data/CR/20200506/20200506_001003.00.002.001_R0.zip",
#         "jobFileId": 1,
#         "storeSource": "minio",
#         "fileId": 3,
#     },
#     {
#         "filePath": r"gzfb/radar_analysis_pic/data/CR/20200506/20200506_001721.00.002.001_R0.zip",
#         "jobFileId": 1,
#         "storeSource": "minio",
#         "fileId": 3,
#     }
# ]
# elevation = 0.5  # [0.5  1.45 2.4  3.35 4.3 ] [5.0 8.0 16.7 30.0 80.0]
# jobid = 2
# # element = "IWC"  # IWCgrid:冰云含量 ,LWCgrid:液体水含量 ,RWCgrid:雨含水量 ,SWCgrid:雪含水量
# point_lat = 24.934353
# point_lon = 119.844055
# title = 'test'
# scan = 1  # 'VOL'
# height = 100  # [100,200,...,2000]
# analysis_type = 1  # 廓线图
#
# for element in ("IWC", "LWC", "RWC", "SWC", "ref", "vel", "sw"):
#     cr = CRController(input_filelist, jobid, element, title, scan, elevation=elevation, height=height, point_lat=point_lat,
#                       point_lon=point_lon, analysis_type=analysis_type)
#     res = cr.run()
#     print(res)
