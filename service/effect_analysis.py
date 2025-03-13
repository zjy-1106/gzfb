# -*- coding: utf-8 -*-
import itertools
import os
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FuncFormatter
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.dates as mdates
from matplotlib.dates import HOURLY, MINUTELY, SECONDLY, DateFormatter, rrulewrapper, RRuleLocator
from metpy.interpolate import cross_section

from config.config import config_info
from service.utils import transfer_path, merge_data, time_delta, get_limit_index, time_gap, ax_add_time_range
from service.colormap import colormap
# from utils.file_uploader import minio_client

# plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False

stat_title = {'max': "max", 'mean': "mean", "area": 'ref ≥', "area50": 'ref <', "custom_area": 'ref = ',
              "state": ''}
legend = ['地物', '湍流', '干雪', '湿雪', '冰晶', '霰', '大滴', '雨', '大雨', '雹']


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    if -2 < b < 2:
        return r'${}$'.format(a * (10 ** b))
    return r'${} · 10^{{{}}}$'.format(a, b)  # \times


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


def set_x_time(ti, tm=None):
    if tm is None:
        lim = get_limit_index(len(ti), num=10)
    else:
        lim = get_limit_index(tm, num=10)
    ts = []
    for i in range(len(ti) - 1):
        ts.append((ti[i + 1] - ti[i]).seconds)
    timestep = max(ts, default=60)
    # timestep = (ti[1] - ti[0]).seconds
    if timestep < 60 and len(ti) < 60:
        if lim > 3:
            lim_m = np.around(lim / 3)
            kwargs_major = dict(freq=MINUTELY, byminute=np.arange(0, 60, 1 * lim_m), bysecond=0)
            lim_m = np.around(lim_m / 2) if (lim_m % 2) == 0 else (np.around(lim_m / 2) - 1)
            kwargs_minor = dict(freq=MINUTELY, byminute=np.arange(0, 60, lim), bysecond=0)
            timestyle = '%H:%M'
        else:
            kwargs_major = dict(freq=SECONDLY, bysecond=np.arange(0, 60, 10 * lim))
            kwargs_minor = dict(freq=SECONDLY, bysecond=np.arange(0, 60, 5 * lim))
            timestyle = '%M′%S″'
    else:
        time_gaps, time_gaps_minor = time_gap(ti)
        timelims_hour = (ti[-1] - ti[0]).total_seconds() / 3600
        if timelims_hour > 16:
            freq_hour = np.concatenate((np.arange(0, ti[0].hour, 3), np.arange(ti[0].hour, 24, 3)))
            kwargs_major = dict(freq=HOURLY, byhour=freq_hour, byminute=0, bysecond=0)
        else:
            kwargs_major = dict(freq=MINUTELY, byminute=np.arange(0, 60, time_gaps), bysecond=0)
        if lim == 1:
            if timestep > 120:
                lim_mint = 15 if int((ti[0] - ti[-1]).seconds / 3600) >= 60 else 6
                kwargs_major = dict(freq=MINUTELY, byminute=np.arange(0, 60, lim_mint), bysecond=0)
                kwargs_minor = dict(freq=MINUTELY, byminute=[ti_m.minute for ti_m in ti], bysecond=0)
            else:
                kwargs_minor = dict(freq=SECONDLY, bysecond=30)
        else:
            lim = np.around(lim / 2) if (lim % 2) == 0 else (np.around(lim / 2) - 1)
            kwargs_minor = dict(freq=MINUTELY, byminute=np.arange(0, 60, time_gaps_minor), bysecond=0)
        timestyle = '%H:%M'
    return kwargs_major, kwargs_minor, timestyle


class Effect_Analysis:

    def __init__(self, fileinfo, jobid, element, title, analysis_type):
        """
        ----------
        param fileinfo:
        param jobid:
        param element: 分析要素
        param title:
        param analysis_type: 统计样式
        param statistic: 分析类目
        ----------
        """
        self.fileinfo = fileinfo
        self.ele = element
        self.title = title
        self.jobid = jobid
        self.analysis_type = analysis_type
        self.statistic = ['max', 'mean']
        pic_file_path = config_info.get_eff_cfg['pic_path']
        self.pic_path = os.path.join(pic_file_path, str(self.jobid))
        if not Path(self.pic_path).exists():
            Path.mkdir(Path(self.pic_path), parents=True, exist_ok=True)

    def get_file_path(self, path):
        data_file_list = []
        for info in self.fileinfo:
            if len(info) >= 2:
                for file_info in info:
                    if file_info.get('radType') == 'HTM':
                        time_str = re.search(r'\d{14}',
                                             os.path.basename(file_info.get('filePath')).replace('_', '')).group()
                        time_day = (datetime.strptime(time_str, '%Y%m%d%H%M%S') - timedelta(hours=8)).strftime('%Y%m%d')
                    else:
                        time_str = re.search(r'\d{14}', os.path.basename(file_info.get('filePath'))).group()  # 源文件为世界时
                        time_day = (datetime.strptime(time_str, '%Y%m%d%H%M%S')).strftime('%Y%m%d')
                    break
                input_file_id = [str(f.get('fileId')) for f in info]
                inp = list(itertools.product(input_file_id, repeat=len(input_file_id)))
                for id_list in inp:
                    if self.analysis_type == 'wind_field':
                        try:
                            data_file_list.append(
                                glob.glob(os.path.join(path, '-'.join(id_list), "DDA*"))[0])  # 文件路径
                            fi = True
                        except IndexError as e:
                            fi = False
                            continue
                        else:
                            break
                    else:
                        try:
                            data_file_list.append(
                                glob.glob(os.path.join(path, time_day, '-'.join(id_list), "*"))[0])  # 文件路径
                            fi = True
                        except IndexError as e:
                            fi = False
                            continue
                        else:
                            break
                if not fi:
                    raise Exception(f'文件未查询成功，请稍后重试。')
            else:
                for file_info in info:
                    input_file_id = str(file_info.get('fileId'))
                    if file_info.get('radType') == 'HTM':
                        time_str = re.search(r'\d{14}',
                                             os.path.basename(file_info.get('filePath')).replace('_', '')).group()
                        time_day = (datetime.strptime(time_str, '%Y%m%d%H%M%S') - timedelta(hours=8)).strftime('%Y%m%d')
                    else:
                        time_str = re.search(r'\d{14}', os.path.basename(file_info.get('filePath'))).group()  # 源文件为世界时
                        time_day = (datetime.strptime(time_str, '%Y%m%d%H%M%S')).strftime('%Y%m%d')
                    try:
                        data_file_list.append(glob.glob(os.path.join(path, time_day, input_file_id, "*"))[0])
                    except IndexError as e:
                        logger.info(f"文件未查询到")
        if not data_file_list:
            raise Exception('文件未查询到')
        return data_file_list

    def plot_broken(self, x, y, xlims, eff_time=None, hail_time=None):
        """
        绘制坐标轴断开图
        :param x: x 数组[]
        :param y: 数据数组
        :param xlims: 断开的位置 (( , ), ( , ))
        :return:
        """
        fig, axis = plt.subplots(1, len(xlims), sharey='all', figsize=(7, 2))
        plt.rcParams['font.size'] = 10
        # x_time = [datetime.strptime(t, '%Y%m%d%H%M%S') for t in x]
        # x = x_time
        if len(xlims) == 1:
            axis = [axis]
        for i, ax in enumerate(axis):
            xlims1, xlims2 = xlims[i]
            if self.ele == 'hcl':
                y = np.array(y).reshape([-1, 10]).T
                for ix in np.arange(10):
                    ax.plot(x, y[ix], marker='.', label=f'{legend[ix]}', color=colormap[self.ele]['colors'][ix])
                    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
                y_min, y_max = ax.get_ylim()
                y_max += y_max / 7
                ax.set_ylim([y_min, y_max])
                ax.legend(loc='upper right', framealpha=0.7, ncol=5)
            elif self.ele == 'rhv':  # 相关系数
                ax.plot(x, y, marker=".")
                ax.set_ylim(0, 1.1)
            else:
                ax.plot(x, y, marker=".")  # , markersize=2
            ax_add_time_range(ax, eff_time, alpha=0.2, color='red')
            ax_add_time_range(ax, hail_time, alpha=0.2, color='purple')

            ax.set_xlim(x[xlims1], x[xlims2])  # 子图1设置y轴范围，只显示部分图
            # ax.set_xticks(x[xlims1:xlims2], labels=[t.strftime("%M:%S") for t in x_time[xlims1:xlims2]])

            # 绘制断裂处的标记
            d = .85  # 设置倾斜度
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15,
                          linestyle='none', color='r', mec='r', mew=1, clip_on=False)
            if i != (len(xlims) - 1):
                ax.spines['right'].set_visible(False)  # 关闭子图1中右部脊
                ax.plot([1, 1], [1, 0], transform=ax.transAxes, markersize=2, **kwargs)
            if i != 0:
                ax.spines['left'].set_visible(False)  # 关闭子图2中左部脊
                ax.plot([0, 0], [0, 1], transform=ax.transAxes, markersize=2, **kwargs)
                ax.tick_params(axis='y', which='both', length=0)  # 关闭y刻度 axis=

            # 绘制主副坐标轴
            if ax.get_ylim()[-1] > 1000:
                ax.yaxis.set_major_formatter(FuncFormatter(fmt))
            kwargs_major, kwargs_minor, timestyle = set_x_time(x)
            majorformatter = DateFormatter(timestyle)
            rule1 = rrulewrapper(**kwargs_major)
            loc1 = RRuleLocator(rule1)
            ax.xaxis.set_major_locator(loc1)
            ax.xaxis.set_major_formatter(majorformatter)
            rule = rrulewrapper(**kwargs_minor)
            loc = RRuleLocator(rule)
            ax.xaxis.set_minor_locator(loc)

            fig.autofmt_xdate()
        return fig, axis

    def get_data_grid(self, str_point, end_point, height, sequence_type=None, statistic_args=None):
        """
        得到绘图的数据：
        单点经纬度数据，则返回对应时间的数据信息列表
        ----------
        param: point_lon: 位置经度点
        param: point_lat: 位置纬度点
        Returns: 字典数据
        -------
        """
        if not statistic_args:
            statistic_args = [50]
        path = config_info.get_eff_cfg['input_dir_grid']
        # path = r"D:\tt"  # T:\renying\grb
        data_file_list = self.get_file_path(path)
        time_list = [datetime.strptime(re.search(r'\d{14}', os.path.basename(file)).group(), '%Y%m%d%H%M%S')
                     + timedelta(hours=8) for file in data_file_list]
        logger.info('开始读取网格数据')
        UNIT_area = 111.194926644 * 0.0009 * 99.029986457 * 0.001
        max_data = []
        mean_data = []
        area_data = []
        area_data50 = []
        hcl_data = []
        color_data = []
        for file in data_file_list:
            with Dataset(file) as ds:
                try:
                    data_ele = np.array(ds.variables[self.ele])
                except KeyError as e:
                    raise Exception(f'没有{e}该要素')
                lon = np.array(ds['Dim3'])
                lat = np.array(ds['Dim2'])
                height_list = list(np.array(ds['Dim1']))
                if self.ele in ['etp', 'crf']:
                    height = [height_list[0], height_list[1]]
                lon_str, lat_str = get_point_index(lon, lat, str_point[0], str_point[1])
                lon_end, lat_end = get_point_index(lon, lat, end_point[0], end_point[1])
                lon_index_str, lon_index_end = min(lon_str, lon_end), max(lon_str, lon_end)
                lat_index_str, lat_index_end = min(lat_str, lat_end), max(lat_str, lat_end)

                if sequence_type == "time_height_change":
                    lon_index_str = lon_index_str - 1 if lon_index_str == lon_index_end else lon_index_str
                    lat_index_str = lat_index_str - 1 if lat_index_str == lat_index_end else lat_index_str
                    color_data = data_ele[:, lat_index_str:lat_index_end, lon_index_str:lon_index_end]
                    color_data[color_data == -999] = None
                    max_d, mean_d = [], []
                    for data_c in color_data:
                        max_d.append(np.nanmax(data_c)) if data_c.size != 0 else max_d.append(np.nan)
                        mean_d.append(np.nanmean(data_c)) if data_c.size != 0 else mean_d.append(np.nan)
                    max_data.append(max_d)
                    mean_data.append(mean_d)
                    data = {"max": max_data, "mean": mean_data, "area": area_data, "state": hcl_data,
                            "area50": area_data50, "colordata": color_data, "height": height_list}
                    continue

                try:
                    idx = []
                    for hh in height:
                        # idx.append(height_list.index(hh))
                        idx.append(np.argmin(abs(np.array(height_list) - hh)))
                except ValueError:
                    raise Exception(f"高度参数错误，无该高度层数据 {hh}")
                if len(idx) == 1:
                    data_ele = data_ele[idx[0], lat_index_str:lat_index_end, lon_index_str:lon_index_end]
                else:
                    data_ele = data_ele[np.arange(idx[0], idx[1] + 1), lat_index_str:lat_index_end,
                                        lon_index_str:lon_index_end]
                data_ele[data_ele == -999] = None
                if self.ele == 'rhv':
                    max_data.append(np.around(np.nanmax(data_ele), 2)) if data_ele.size != 0 else max_data.append(0)
                else:
                    max_data.append(np.nanmax(data_ele)) if data_ele.size != 0 else max_data.append(0)
                mean_data.append(np.nanmean(data_ele)) if data_ele.size != 0 else mean_data.append(0)
                area_data.append(np.sum(data_ele >= statistic_args[0])) if data_ele.size != 0 else area_data.append(0)
                area_data50.append(np.sum(data_ele < statistic_args[-1])) if data_ele.size != 0 else area_data50.append(0)
                area_data25 = abs(np.array(area_data) - np.array(area_data50))
                [hcl_data.append(np.sum(data_ele == n)) for n in np.arange(10)]
                data = {"max": max_data, "mean": mean_data, "area": area_data, "state": hcl_data, "area50": area_data50,
                        "custom_area": area_data25, "colordata": color_data, "height": height_list}

        return data, time_list

    def get_data_wind(self):
        """
        得到三维风绘图的数据：
        ----------
        Returns: 字典数据
        -------
        """

        path = config_info.get_eff_cfg['input_dir_wind']
        # path = r"D:\tt"  # T:\Product\dda_wind
        data_file_list = self.get_file_path(path)
        time_list = [datetime.strptime(re.search(r'\d{14}', os.path.basename(file)).group(), '%Y%m%d%H%M%S')
                     + timedelta(hours=8) for file in data_file_list]
        logger.info('开始读取三维风数据')
        for file in data_file_list:
            with xr.open_dataset(file) as ds:
                lat = ds.variables['point_latitude'][0, :, 0]
                lon = ds.variables['point_longitude'][0, 0, :]
                height = ds.variables['z'] / 1000
                ref = ds.variables['reflectivity'][0]
                u = ds.variables['u'][0]
                v = ds.variables['v'][0]
                w = ds.variables['w'][0]

                section_data = xr.Dataset(data_vars={'u': (['height', 'lat', 'lon'], u),
                                                     'v': (['height', 'lat', 'lon'], v),
                                                     'w': (['height', 'lat', 'lon'], w),
                                                     'ref': (['height', 'lat', 'lon'], ref)},
                                          coords={'lon': (['lon'], lon),
                                                  'lat': (['lat'], lat),
                                                  'height': (['height'], height)}
                                          )
                section_data = section_data.metpy.parse_cf().squeeze()

        return section_data, time_list

    def get_data_polar(self, height, azimuth, statistic_args=None):
        """
        得到绘图的数据：
        若没有单点经纬度数据，则返回对应时间的数据信息列表
        若输入所选取的经纬度点的位置，返回单点的风场数据，绘制单点时间高度图。
        ----------
        param: height: 高度 m
        param: azimuth: 方位角
        Returns: 字典数据
        -------
        """
        if statistic_args is None:
            statistic_args = [50]
        height = [i / 1000 for i in height]
        path = config_info.get_eff_cfg['input_dir_polar']
        # path = r"D:\tt"  # T:\renying\pol
        data_file_list = self.get_file_path(path)
        time_list = [datetime.strptime(re.search(r'\d{14}', os.path.basename(file)).group(), '%Y%m%d%H%M%S')
                     + timedelta(hours=8) for file in data_file_list]
        logger.info('开始读取极坐标数据')
        UNIT_area = 111.194926644 * 0.0009 * 99.029986457 * 0.001
        max_data = []
        mean_data = []
        area_data = []
        area_data50 = []
        for file in data_file_list:
            with xr.open_dataset(file) as ds:
                data_ele = ds.variables[self.ele].values
                dim1 = ds.variables['Dim1'][:].values
                dim2 = ds.variables['Dim2'][:].values
                dim3 = ds.variables['Dim3'][:].values
                deg2rad = np.pi / 180
                # 筛选方位角
                if azimuth:
                    idx1 = np.argmin(np.abs(np.array(dim2) - azimuth[0]))
                    idx2 = np.argmin(np.abs(np.array(dim2) - azimuth[1]))
                else:
                    idx1 = 0
                    idx2 = len(dim2)
                # 筛选高度
                y = dim3.reshape(len(dim3), 1) * np.sin(dim1.reshape(1, len(dim1)) * deg2rad) / 1000
                data_temp = np.full_like(data_ele, np.nan)
                for i in range(y.shape[1]):
                    low = np.where(y[:, i] > height[0])[0]
                    high = np.where(y[:, i] < height[1])[0]
                    if low.shape[0] == 0:
                        continue
                    if low[0] == high[-1]:
                        if idx1 == idx2:
                            data_temp[i, idx1, low[0]] = data_ele[i, idx1, low[0]]
                        else:
                            data_temp[i, idx1:idx2, low[0]] = data_ele[i, idx1:idx2, low[0]]
                    else:
                        if idx1 == idx2:
                            data_temp[i, idx1, low[0]:high[-1]] = data_ele[i, idx1, low[0]:high[-1]]
                        else:
                            data_temp[i, idx1:idx2, low[0]:high[-1]] = data_ele[i, idx1:idx2, low[0]:high[-1]]
                data_temp[data_temp == -999] = None
                max_data.append(np.nanmax(data_temp))
                mean_data.append(np.nanmean(data_temp))
                area_data.append(np.sum(data_temp >= statistic_args[0]))
                area_data50.append(np.sum(data_temp < statistic_args[-1]))
                area_data25 = abs(np.array(area_data) - np.array(area_data50))
                data = {"max": max_data, "mean": mean_data, "area": area_data, "area50": area_data50,
                        "custom_area": area_data25}

        return data, time_list

    def draw_rad_line(self, time_list, data, eff_time, statistic, title, statistic_args=None, hail_time=None):
        """
        时间段的统计分析图
        ---------
        param time_list: 时间序列列表
        param data: 数据
        param str_time: 起始时间
        param end_time: 结束时间
        return: 图片信息
        """
        if statistic_args is None:
            statistic_args = [50]
        # fig, ax = plt.subplots()
        try:
            scan_type = os.path.basename(self.fileinfo[0][0]['filePath']).split('-')[2]
        except IndexError:
            scan_type = 'CAP'

        if -9999 in statistic_args:
            data = data["area50"]
        elif 9999 in statistic_args:
            data = data["area"]
        else:
            data = data[statistic]

        if scan_type == 'SEC':
            tti = time_delta(time_list)
            tti = [(0, -1)]
            fig, axis = self.plot_broken(time_list, data, tti, eff_time, hail_time=hail_time)
            plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=None)
            if 'area' in statistic:
                if len(statistic_args) == 1:
                    sta_title = f'{stat_title[statistic].capitalize()}{statistic_args[0]}dBZ'
                elif -9999 in statistic_args:
                    sta_title = f'{stat_title[statistic].capitalize()}<{statistic_args[-1]}dBZ'
                elif 9999 in statistic_args:
                    sta_title = f'{stat_title[statistic].capitalize()}≥{statistic_args[0]}dBZ'
                else:
                    sta_title = f'{stat_title[statistic]}{statistic_args[0]}~{statistic_args[-1]}dBZ'
                axis[0].text(0.1, 1.04, f'{title}', fontsize=10, transform=axis[0].transAxes)
                axis[0].text(0.4, 1.13, f'{self.title}  {sta_title}', fontsize=12, transform=axis[0].transAxes)
                # plt.suptitle(f'{self.title}  {sta_title}', fontsize=12)
                # axis[0].set_title(f'{title}', fontsize=10)
            else:
                axis[0].text(0.1, 1.04, f'{title}', fontsize=10, transform=axis[0].transAxes)
                axis[0].text(0.4, 1.13, f'{self.title}  {stat_title[statistic].capitalize()}  {self.ele.capitalize()}',
                             fontsize=12, transform=axis[0].transAxes)
                # plt.suptitle(f'{self.title}  {stat_title[statistic].capitalize()}  {self.ele.capitalize()}', fontsize=12)
                # axis[0].set_title(
                #     f'{title}', fontsize=10)
            if "area" in statistic:
                axis[0].set_ylabel(' ')
            else:
                axis[0].set_ylabel(colormap[self.ele]["label"])
        elif scan_type == 'CAP':
            # time_range = pd.date_range(time_list[0], time_list[-1], freq='min')
            # df1 = pd.DataFrame(columns=['data_time'], data=np.array(time_range))
            # df2 = pd.DataFrame({'data_time': time_list, 'data': data})
            # new_df = pd.merge(left=df1, right=df2, how='left', on='data_time')
            tti = time_delta(time_list)
            tti = [(0, -1)]

            # plt.plot(time_range, np.around(new_df.data, 3), marker=".")
            fig, axis = self.plot_broken(time_list, data, tti, eff_time, hail_time=hail_time)
            plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.2, hspace=None)
            if 'area' in statistic:
                if len(statistic_args) == 1:
                    sta_title = f'{stat_title[statistic].capitalize()}{statistic_args[0]}dBZ'
                elif -9999 in statistic_args:
                    sta_title = f'{stat_title[statistic].capitalize()}<{statistic_args[-1]}dBZ'
                elif 9999 in statistic_args:
                    sta_title = f'{stat_title[statistic].capitalize()}≥{statistic_args[0]}dBZ'
                else:
                    sta_title = f'{stat_title[statistic]}{statistic_args[0]}~{statistic_args[-1]}dBZ'
                axis[0].text(0.1, 1.04, f'{title}', fontsize=10, transform=axis[0].transAxes)
                axis[0].text(0.4, 1.13, f'{self.title}  {sta_title}', fontsize=12, transform=axis[0].transAxes)
                # plt.suptitle(f'{self.title}  {sta_title}', fontsize=12)
                # axis[0].set_title(f'{title}', fontsize=10)
            else:
                axis[0].text(0.1, 1.04, f'{title}', fontsize=10, transform=axis[0].transAxes)
                axis[0].text(0.4, 1.13, f'{self.title}  {stat_title[statistic].capitalize()}  {self.ele.capitalize()}',
                             fontsize=12, transform=axis[0].transAxes)
                # plt.suptitle(f'{self.title}  {stat_title[statistic].capitalize()}  {self.ele.capitalize()}', fontsize=12)
                # axis[0].set_title(
                #     f'{title}', fontsize=10)

            if "area" in statistic:
                axis[0].set_ylabel(' ')
            else:
                axis[0].set_ylabel(colormap[self.ele]["label"])
        else:
            raise TypeError('雷达文件扫描模式错误')

        png_path = os.path.join(self.pic_path,
                                f"{time_list[0].strftime('%Y%m%d%H%M%S')}-{time_list[-1].strftime('%Y%m%d%H%M%S')}"
                                f"_{self.analysis_type}_{self.ele}_{statistic}.png")
        plt.savefig(png_path, dpi=300, transparent=False, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": self.ele}
        logger.info(f"生成图片：{png_path}")
        return pic_info

    def draw_ref_wind(self, section_data, eles, sta_point, end_point, time_list):
        """
        任意两点间的垂直剖面图
        ----------
        param section_data: 数据信息
        param sta_point: 起始点
        param end_point: 结束点
        return:
        ----------
        """

        start = sta_point[::-1]
        end = end_point[::-1]
        cross = cross_section(section_data, start, end)
        fig = plt.figure(figsize=(13, 5))
        i = 0
        for ele in eles:
            i = i + 1
            plt.subplot(1, 2, i)
            if start[0] == end[0]:
                x = cross['lon']
            else:
                x = cross['lat']

            cmap = ListedColormap(colormap[ele]["colors"])
            norm = BoundaryNorm(boundaries=colormap[ele]["levels"], ncolors=len(colormap[ele]["colors"]))
            plt.pcolormesh(x, cross['height'], cross[ele],
                           cmap=cmap, norm=norm, shading='auto', zorder=0)
            barb_density_y = 1
            barb_density_x = 4
            barb_len = 5
            barb_line_width = 0.5
            plt.barbs(x[::barb_density_x], cross['height'][::barb_density_y],
                      cross['u'][::barb_density_y, ::barb_density_x], cross['v'][::barb_density_y, ::barb_density_x],
                      length=barb_len, linewidth=barb_line_width,
                      sizes={"spacing": 0.2, 'height': 0.5, 'width': 0.5, 'emptybarb': 0}, zorder=1,
                      barb_increments=dict(half=2, full=4, flag=20))

            plt.title(self.title + f'({colormap[ele]["title"]})')
            br = plt.colorbar(fraction=0.027, pad=0.035, ticks=colormap[ele]["levels"], aspect=30)
            br.ax.set_title(colormap[ele]['label'])

            if start[1] > end[1]:
                lons = np.arange(end[1], start[1] + 0.0001, (start[1] - end[1]) / 4)
                lons = np.flipud(lons)
            elif start[1] < end[1]:
                lons = np.arange(start[1], end[1] + 0.0001, (end[1] - start[1]) / 4)
            else:
                lons = np.full(5, start[1])

            if start[0] > end[0]:
                ax = plt.gca()
                ax.invert_xaxis()
                lats = np.arange(end[0], start[0] + 0.0001, (start[0] - end[0]) / 4)
                lats = np.insert(lats, 0, 0, axis=0)
                lons = np.insert(lons, 0, 0, axis=0)
                plt.xticks(ticks=lats[1:6],
                           labels=[f'{round(lons[-i], 2)}°E \n{round(lats[i], 2)}°N' for i in range(1, 6)])
            elif start[0] < end[0]:
                lats = np.arange(start[0], end[0] + 0.0001, (end[0] - start[0]) / 4)
                plt.xticks(ticks=lats, labels=[f'{round(lons[i], 2)}°E \n{round(lats[i], 2)}°N' for i in range(5)])
            else:
                lats = np.full(5, start[0])
                plt.xticks(ticks=lons, labels=[f'{round(lons[i], 2)}°E \n{round(lats[i], 2)}°N' for i in range(5)])
            plt.ylim([0, cross['height'][-1] + 1])
            plt.ylabel('高度(km)')
        png_path = os.path.join(self.pic_path, f"{time_list[0].strftime('%Y%m%d%H%M%S')}_{self.ele}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": self.ele}
        logger.info(f"生成图片：{png_path}")
        return pic_info

    def draw_time_point(self, time_list, data, statistic, eff_time=None, hail_time=None):
        """
        时空分析图
        ---------
        param time_list: 时间序列列表
        param data: 数据
        return: 图片信息
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.rcParams['font.size'] = 12
        try:
            color_data = np.array(data[statistic], dtype=np.float)
        except AttributeError:
            color_data = np.array(data[statistic])
        color_data[color_data == -999] = None
        time_x = time_list  # [1,2,3,4]
        cmap = ListedColormap(colormap[self.ele]["colors"])
        norm = BoundaryNorm(boundaries=colormap[self.ele]["levels"], ncolors=len(colormap[self.ele]["colors"]))
        x, y = np.meshgrid(time_x, np.array(data['height']) / 1000)
        im = plt.pcolormesh(x, y, color_data.T,
                            cmap=cmap, norm=norm, shading='auto', zorder=0)
        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

        if self.ele == 'scw':
            bins = [1, 2, 3]
            nbin = len(bins) - 1
            cmap = ListedColormap(['#a0a0a0', '#ff0000'])
            norm4 = BoundaryNorm(bins, nbin)
            im4 = matplotlib.cm.ScalarMappable(norm=norm4, cmap=cmap)
            cb = fig.colorbar(im4, ax=ax)
        else:
            cb = fig.colorbar(im, ticks=colormap[self.ele].get('levels'), fraction=0.03, pad=0.035, aspect=30)
            cb.ax.set_title(colormap[self.ele]['label'])  # 设置单位
        if self.ele == 'hcl':
            # br = plt.colorbar(cax=position, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            cb.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            cb.set_ticklabels(['地物', '湍流', '干雪', '湿雪', '冰晶', '霰', '大滴', '雨', '大雨', '雹'])
        elif self.ele == 'scw':
            cb.set_ticks([1.5, 2.5])
            cb.set_ticklabels(['非过冷水', '过冷水'])
        elif self.ele == 'ccl':
            cb.set_ticks([0.5, 1.5])
            cb.set_ticklabels(['对流云', '层状云'])

        kwargs_major, kwargs_minor, timestyle = set_x_time(time_list)
        majorformatter = DateFormatter(timestyle)
        rule1 = mdates.rrulewrapper(**kwargs_major)
        loc1 = mdates.RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = mdates.rrulewrapper(**kwargs_minor)
        loc = mdates.RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)
        # lim = 1
        # if len(time_list) > 10:
        #     lim = round(len(time_list) / 10)
        # ax.set_xticks(time_x[::lim], labels=[t.strftime("%H:%M") for t in time_list][::lim])
        # plt.xlim(time_list[0], time_list[-1])
        fig.autofmt_xdate(rotation=45)  # rotation=45
        ax.set_ylim([2.5, y[-1, -1]])
        plt.ylabel('海拔高度（km）')
        # plt.title(self.title)
        plt.title(f'{self.title} {stat_title[statistic].capitalize()} {self.ele.capitalize()}', fontsize=15)
        png_path = os.path.join(self.pic_path,
                                f"{time_list[0].strftime('%Y%m%d%H%M%S')}-{time_list[-1].strftime('%Y%m%d%H%M%S')}"
                                f"_{self.ele}_time_height_{statistic}.png")
        plt.savefig(png_path, dpi=400, transparent=False, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": self.ele}
        logger.info(f"生成图片：{png_path}")
        return pic_info

    def run(self, str_point=None, end_point=None, eff_time=None, hail_time=None, height=None, azimuth=None, sequence_type=None,
            statistic_args=None):
        """
        格点化图：始末点，始末时间
        极坐标图：高度以及方位角范围，始末时间
        风场分析图：始末点
        -----------
        param str_point: 格点化区域的起始点
        param end_point: 格点化区域的结束点
        param str_time: 作业起始时间
        param end_time: 作业结束时间
        param height: 极坐标区域高度范围
        param azimuth: 极坐标区域方位角范围
        param sequence_type: 统计类型
        param statistic_args: 统计自定义区间
        return:
        ------------
        """
        if not statistic_args:
            statistic_args = [50]
        sequence_type = "time_sequence_change" if not sequence_type else sequence_type
        info_pic = []
        if self.ele == 'ref':
            self.statistic.extend(['custom_area'])
        elif self.ele == 'hcl':
            self.statistic = ['state']

        if self.analysis_type == 'color':
            data, time_list = self.get_data_grid(str_point, end_point, height, sequence_type, statistic_args)
            title_time = '~'.join({time_list[0].strftime("%Y-%m-%d"), time_list[-1].strftime("%Y-%m-%d")})
            if len(time_list) <= 1:
                raise Exception('数据时次小于2，不足绘制填色图')
            logger.info("开始绘制网格统计图")
            if sequence_type == "time_height_change":  # 绘制时空分布图
                if len(time_list) <= 1:
                    raise Exception('数据时次小于2，不足绘制填色图')
                for statistic in ["max", "mean"]:
                    info_pic.append(self.draw_time_point(time_list, data, statistic, eff_time=eff_time, hail_time=hail_time))
            else:
                height = [np.around(i / 1000, 1) for i in height]
                title = f'{title_time}  ' \
                        f's={np.around(str_point[0], 2)}°E, {np.around(str_point[-1], 2)}°N' \
                        f'~{np.around(end_point[0], 2)}°E, {np.around(end_point[-1], 2)}°N  ' \
                        f'z={"~".join(list(map(str, height)))}km'
                title = title.split('z')[0] if self.ele in ['etp', 'crf'] else title
                for statistic in self.statistic:
                    if len(statistic_args) > 1:
                        info_pic.append(self.draw_rad_line(time_list, data, eff_time, 'custom_area', title, statistic_args, hail_time=hail_time))
                        break
                    info_pic.append(self.draw_rad_line(time_list, data, eff_time, statistic, title, statistic_args, hail_time=hail_time))
        elif self.analysis_type == 'wind_field':
            data, time_list = self.get_data_wind()
            if self.ele == 'wind_st':
                logger.info("开始绘制风场分析图")
                info_pic.append(self.draw_ref_wind(data, ['ref', 'w'], str_point, end_point, time_list))
            else:
                raise KeyError(f'请输入正确element, element={self.ele}')
        elif self.analysis_type == 'polar':
            data, time_list = self.get_data_polar(height, azimuth, statistic_args)
            height = [np.around(i / 1000, 1) for i in height]
            title_time = '~'.join({time_list[0].strftime("%Y-%m-%d"), time_list[-1].strftime("%Y-%m-%d")})
            title = f'{title_time} z={"~".join(list(map(str, height)))}km azm={"~".join(list(map(str, azimuth)))}'
            title = ' '.join(title.split(' ')[::2]) if self.ele == 'etp' else title
            logger.info("开始绘制极坐标统计图")
            if len(time_list) <= 1:
                raise Exception('数据时次小于2，不足绘制填色图')
            for statistic in self.statistic:
                if len(statistic_args) > 1:
                    info_pic.append(self.draw_rad_line(time_list, data, eff_time, 'custom_area', title,
                                                       statistic_args, hail_time=hail_time))
                    break
                info_pic.append(self.draw_rad_line(time_list, data, eff_time, statistic, title, statistic_args, hail_time=hail_time))
        else:
            raise ValueError(f"analysis_type={self.analysis_type},请输入正确的分析类型")

        return [{"picFiles": info_pic}]

# if __name__ == '__main__':
#     jobid = 2020
#     fileinfo_wind = [[{
#         "fileId": 1,
#         "filePath": "D:/work/test/wind_3d/wind/out/temp/DDA_70706_ZWN01_20220727055305.nc",
#         "radType": "ETW",
#         "storeSource": "/minio"
#     }]]
#     fileinfo_pro = [[{
#         "fileId": 2,
#         "filePath": "D:/work/test/wind_3d/wind/out/temp/DDA_70706_ZWN01_20220727055305.nc",
#         "radType": "ETW",
#         "storeSource": "/minio"
#     }]]
#     fileinfo = [[
#         {"fileId": 290028,
#          "filePath": "gzfb/renying/origin_data/hebian/20220727/VOL/20220727135305.00.002.001_R1.zip",
#          "radType": "ETW",
#          "storeSource": "/minio"
#          }],
#         [{"fileId": 290025,
#           "filePath": "gzfb/renying/origin_data/hebian/20220727/VOL/20220727135305.00.002.001_R1.zip",
#           "radType": "ETW",
#           "storeSource": "/minio"
#           }],
#         [{"fileId": 290027,
#           "filePath": "gzfb/renying/origin_data/ZWN01/20220727/Z_RADR_I_ZWN01_20220727055400_O_DOR-XPD-CAP-FMT.BIN.zip",
#           "radType": "ETW",
#           "storeSource": "/minio"}],
#         [{"fileId": 283359,
#           "filePath": "gzfb/renying/origin_data/hebian/20220727/VOL/20220727135305.00.002.001_R1.zip",
#           "radType": "ETW",
#           "storeSource": "/minio"
#           }]
#     ]
#     # id = [f.get("fileId") for f in fileinfo]
#
#     element = 'ref'
#     tic = 'max'
#     title = ' '
#     str_p = [104.15, 27.04]
#     end = [104.24, 26.9]
#     h = [1, 5]
#     amu = [110, 140]
#     analysis_type = "grid"  # "wind"  # "rad"
#     # mid = [104.12, 27.05]
#     mid = None
#     p = Effect_Analysis(fileinfo, jobid, element, title, analysis_type, statistic=tic, level=40)
#
#     p.run(str_p, end, ['20220727141000', '20220727144000'], ['20220727143000', '20220727145000'], h, amu)
