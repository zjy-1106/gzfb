#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import itertools
import re
import time
from datetime import datetime, timedelta

import os
from math import sin, cos, atan2, pi
from pathlib import Path

import geopy
import h5py
import numpy as np
import pandas as pd
import matplotlib
from loguru import logger

from config.config import config_info
import standard.standard_latlon_lib as StandardToLatLon
# from utils.file_uploader import minio_client


def transfer_path(path_str, is_win_path=False, is_inline_path=False, jobid=None):
    """
    转换windows与minio挂载路径
    :param path_str: 传入路径
    :param is_win_path: 是否是win路径
    :param is_inline_path: 是否替换挂载路径
    :return:
    """
    path_cfg = config_info.get_path
    minio_backname = path_cfg.get('minio_disk')
    if jobid:
        path_jobid = Path(path_str).parts[-2]
        if str(jobid) == path_jobid:
            temp_path = list(Path(path_str).parts[1:])
            del temp_path[-2]
            path_str = '/'.join(temp_path)
        else:
            path_str = '/'.join(Path(path_str).parts[1:])
    if is_win_path:
        if is_inline_path:
            return path_str.replace(path_cfg['file_save_path'][0:2], path_cfg['disk_path'][0:2]).replace('\\', '/')
        return path_str.replace(path_cfg['disk_path'], minio_backname).replace('\\', '/')
    if path_str[0] == '/':
        path_str = path_str[1::]
    filepath = os.path.join(path_cfg['disk_path'], path_str).replace('\\', '/')
    return filepath


def get_lon_lat(distance, azimuth, elevation, centerlon, centerlat, h_offset=True):
    """
    获取数据所有点的经纬度
    :param distance: 径向距离库数
    :param azimuth: 方位角数
    :param elevation: 海拔高度
    :param centerlon: 雷达经度
    :param centerlat: 雷达纬度
    :param h_offset:
    :return:
    """
    # 返回千米单位，用于插值到一公里的数据 deltav, deltah
    elev = elevation if h_offset else 0
    deg2rad = np.pi / 180
    azimuth = azimuth * deg2rad
    if isinstance(azimuth, np.ndarray):
        deltav = np.cos(azimuth[:, np.newaxis]) * distance * np.cos(elev * deg2rad)
        deltah = np.sin(azimuth[:, np.newaxis]) * distance * np.cos(elev * deg2rad)
    else:
        deltav = np.cos(azimuth) * distance * np.cos(elev * deg2rad)
        deltah = np.sin(azimuth) * distance * np.cos(elev * deg2rad)
    deltalat = deltav / 111
    actuallat = deltalat + centerlat
    deltalon = deltah / (111 * np.cos(actuallat * deg2rad))
    actuallon = deltalon + centerlon
    return deltav, deltah, actuallon, actuallat


def get_point_data(lat, lon, data, point_lat, point_lon):
    # 获取某个经纬度点的格点数据
    # 数据包含高度层
    # lat_idx = np.unravel_index(np.argmin(np.abs(lat - point_lat), axis=None), data.shape)[1]
    # lon_idx = np.unravel_index(np.argmin(np.abs(lon - point_lon), axis=None), data.shape)[2]
    lat_idx = np.argmin(np.abs(lat - point_lat), axis=None)
    lon_idx = np.argmin(np.abs(lon - point_lon), axis=None)
    return data[:, lat_idx, lon_idx]


def get_limit_index(list_len, num=10):
    """
    稀疏列表
    :param list_len: 目标列表长度
    :param num: 稀疏后列表最大长度（默认10）
    :return: step 稀疏后步长
    """
    return 1 if list_len < num else int(round(list_len / num, 0))


def merge_data(time_range1, time_range2, data):
    # 合并两个dataframe

    df1 = pd.DataFrame(columns=['data_time'], data=np.array(time_range1))

    data_array = np.array([time_range2] + np.array(data).T.tolist(), dtype=object).T
    data_list = [f'data{i}' for i in range(len(data[0]))]
    df2 = pd.DataFrame(columns=['data_time'] + data_list, data=data_array)

    new_df = pd.merge(left=df1, right=df2, how='left', on='data_time')

    return new_df


def temp_message(temp_file, fileid_list, jobid=None):
    """
    生产中间文件消息
    :param temp_file: 中间文件路径列表
    :param fileid_list: 对于文件id
    :return: 中间文件信息
    """
    if type(temp_file) == str:
        temp_file_info = [
            {"path": transfer_path(temp_file, is_win_path=True, jobid=jobid),
             "filename": os.path.basename(temp_file),
             "fileSize": os.path.getsize(temp_file),
             "fileIds": fileid_list}
        ]
    else:
        temp_file_info = [
            {"path": transfer_path(f, is_win_path=True, jobid=jobid),
             "filename": os.path.basename(f),
             "fileSize": os.path.getsize(f),
             "fileIds": fileid_list} for f in temp_file
        ]
    # try:
    #     temp_file_info = [
    #             {"path": transfer_path(f, is_win_path=True),
    #              "filename": os.path.basename(f),
    #              "fileSize": os.path.getsize(f),
    #              "fileIds": fileid_list} for f in temp_file
    #         ]
    # except AttributeError:
    #     temp_file_info = [
    #         {"path": transfer_path(f[0], is_win_path=True),
    #          "filename": os.path.basename(f[0]),
    #          "fileSize": os.path.getsize(f[0]),
    #          "fileIds": fileid_list} for f in temp_file
    #     ]
    return [{"tempFiles": temp_file_info}]


def select_data(data, ele, threshold):
    threshold_low, threshold_high = threshold  # 阈值参数改为列表，[最小值，最大值]
    # threshold_low = threshold_low if threshold_low != colormap[ele]['levels'][0] else float('-inf')  # 最小值
    data[data < threshold_low] = None
    data[data > threshold_high] = None
    return data


def time_delta(times):
    lims = [-1]
    if 'datetime.datetime' in str(type(times[0])):
        ti = times
    else:
        ti = [datetime.strptime(t, '%Y%m%d%H%M%S') for t in times]
    for i in range(len(ti) - 1):
        if (ti[i + 1] - ti[i]).seconds > 100:  # 设置多少时间间隔断开
            lims.append(i)
            # lims.append(i+1)
    lims.append(len(ti) - 1)
    lims_list = []
    for k in range(len(lims) - 1):
        lims_list.append((lims[k] + 1, lims[k + 1]))

    return lims_list


def prj_nom(obi_full_path, nom_full_path, ele):
    """
    处理卫星原始观测与等经纬投影的转换方法
    :param obi_full_path:行列号转等经纬投影的静态查找表文件
    :param nom_full_path: 原始观测文件
    :param ele: 观测要素
    :return: 投影后数据
    """
    s_line = 175  # 云图起始行号
    length = 1092  # 云图高度
    with h5py.File(obi_full_path, 'r') as f:
        lat = f['Lat'][s_line:s_line + length]
        lon = f['Lon'][s_line:s_line + length]

    with h5py.File(nom_full_path, 'r') as f:
        if 'REGC' in nom_full_path:
            data = f[ele][...]
        else:
            data = f[ele][s_line:s_line + length]

    strandard = StandardToLatLon.StandardToLatLon(5, 54, 60, 139, 65535, resolution=0.04)
    res_list = strandard.proj_process(lon, lat, data)

    return res_list


def wgs84toWebMercator(lon, lat):
    """
    将经纬度转为web墨卡托下的坐标,转换公式如下
    :param lon: 经度,需传入numpy数组,范围为[-180,180]
    :param lat: 纬度,需传入numpy数组,范围为[-85.05，85.05]
    :return:x,横坐标,对应于经度;y,纵坐标,对应于纬度
    """
    x = lon * 20037508.342789 / 180
    y = np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180)
    y = y * 20037508.34789 / 180
    return x, y


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


def extend_point(start_lat, start_lon, end_lat, end_lon, distance=5):
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


def file_exit(filepath, sleep_times=5):
    """
    :param filepath: 文件minio所在路径
    :param sleep_times: 循环次数
    """
    if os.path.isfile(transfer_path(filepath)):
        return transfer_path(filepath)
    raise NameError('文件本地地址错误')


def list_to_array(x, fill_value=-999, dtype=int):
    """
    不规则list转array
    :param x: 列表
    :param fill_value: 空值填充值
    :param dtype: array数据类型
    :return:
    """
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(x)], axis=1)
    return dff.fillna(fill_value).values.T.astype(dtype)


def upload_files(local_path, jobid=None):
    path_cfg = config_info.get_path
    # for path in mid_path:
    local_path_disk = path_cfg['file_save_path']
    if type(local_path) not in [list, str]:
        raise TypeError('仅支持列表文件名或者字符串')
    try:
        local_path = os.path.dirname(local_path) if os.path.isfile(local_path) else local_path
        for dir_path, dir_names, file_names in os.walk(local_path):
            if file_names:
                for file_name in file_names:
                    if dir_path[0:2] in local_path_disk[0:2]:
                        srt_t = time.time()
                        local_filepath = os.path.join(dir_path, file_name)
                        # minio_client.put_object(local_filepath, jobid=jobid)
                        logger.info(f'文件{local_filepath}上传，耗时{time.time() - srt_t}')
                        # os.remove(local_filepath)
    except TypeError:
        for local_p in local_path:
            upload_files(local_p, jobid)


def del_rmdir(dirpath):
    dir_list = []
    for root, dirs, files in os.walk(dirpath):
        dir_list.append(root)
    # 先生成文件夹的列表，重点是下边
    for root in dir_list[::-1]:
        if not os.listdir(root):
            try:
                os.rmdir(root)
            except OSError:
                return


def time_gap(times):
    timelims_hour = (times[-1] - times[0]).total_seconds() / 3600
    if timelims_hour <= 0.25:
        return 1, 1
    elif timelims_hour <= 1:
        return 5, 1
    elif timelims_hour < 4:
        return 15, 5
    elif timelims_hour < 8:
        return 30, 10
    elif timelims_hour < 16:
        return 60, 20
    elif timelims_hour < 24:
        return 60, 60
    else:
        return 60, 60


def ax_add_time_range(ax, range_time, ylims=None, alpha=None, color=None):
    color_name = {'w': "白色", 'k': "黑色", 'red': "红色", 'purple': '紫色'}
    if color:
        c = color
    else:
        c = 'red'
    if alpha:
        alp = alpha
    else:
        alp = 0.3
    if range_time:
        for k in range(len(range_time)):
            time1 = datetime.strptime(range_time[k][0], '%Y%m%d%H%M%S')
            time2 = datetime.strptime(range_time[k][1], '%Y%m%d%H%M%S')
            if ylims is not None:
                ax.vlines([time1, time2], ymin=ylims[0], ymax=ylims[-1], linestyles='dashed', colors=c)
            else:
                ax.axvspan(time1, time2, alpha=alp, color=c)
        if color in ['w', 'red']:
            ax.text(0.01, 0.9, f'{color_name[color]}：作业时间;', transform=ax.transAxes)
        else:
            ax.text(0.2, 0.9, f'{color_name[color]}：降雹时间', transform=ax.transAxes)


def get_file_path(fileinfo, path, img_type='ref'):
    data_file_list = []
    if len(fileinfo) == 1:
        fileinfo = [fileinfo]
    for info in fileinfo:
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
                if img_type:
                    if 'wind' in img_type:
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


def add_right_cax(ax, pad, width):
    """
    在一个ax右边追加与之等高的cax.
    ----------
    param: ax:
    param: pad: 是cax与ax的间距.
    param: width: 是cax的宽度.
    ----------
    """
    axpos = ax.get_position()
    caxpos = matplotlib.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax