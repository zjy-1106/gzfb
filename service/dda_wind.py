#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三维风场程序，输入雷达质控前nc文件，输出三维风场nc文件和质控后的雷达nc文件
"""
import os
import glob
import re
import time

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
from pyart.io import read_grid
import numpy as np
import xarray as xr
from geopy.distance import geodesic
from matplotlib import pyplot as plt
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.dates import DateFormatter, rrulewrapper, RRuleLocator
from metpy.interpolate import cross_section
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl

from config.config import config_info
from mq.pubulisher import publish_temp_msg
from service.effect_analysis import set_x_time
from service.utils import transfer_path, temp_message, select_data, get_limit_index, extend_point, file_exit, \
     ax_add_time_range, add_right_cax
from service.colormap import colormap
# from utils.file_uploader import minio_client

matplotlib.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
mpl.use('Agg')
FONTSIZE = 12
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


class DDAWindController:
    def __init__(self, fileinfo, level, jobid, title=None, threshold=None):
        """
        ----------
        param: fileinfo: 传入的文件信息
        param: level: 绘制水平风场的层高
        param: jobid:
        ----------
        """

        self.fileinfo = fileinfo
        self.level = int(level / 1000) - 1 if level else 0
        self.jobid = jobid
        self.title = title
        self.threshold = threshold
        self.height_list = []
        self.path_cfg = config_info.get_dda_wind_config
        self.pic_file_path = self.path_cfg["pic_path"] + '/' + str(self.jobid)  # 生成图片保存的路径
        self.exe_path = self.path_cfg["exe_path"]  # os.path.join(BASE_PATH, 'exe', 'DDA_wind', 'DDAProcessNC.exe') exe路径
        self.listfile = self.run_exe()
        self.listfile = sorted(self.listfile, key=lambda times: re.search(r'\d{14}', times).group())
        self.time_index = [(datetime.strptime(re.search(r"\d{14}", t).group(), '%Y%m%d%H%M%S') + timedelta(hours=8)
                            ).strftime('%Y%m%d%H%M%S') for t in self.listfile]  # 绘制图片的时间序列

    def run_exe(self):

        listfile = []
        for file_info in self.fileinfo:
            file_info = sorted(file_info, key=lambda x: x.get("equipNum"))
            input_file_id = [file.get('fileId') for file in file_info]
            file_label = '-'.join([str(file['fileId']) for file in file_info])
            if file_info[0].get('radType') == 'HTM':
                time_str = re.search(r'\d{8}_\d{2}', os.path.basename(file_info[0].get('filePath'))).group().replace(
                    '_', '')
                time_str = (datetime.strptime(time_str, '%Y%m%d%H') - timedelta(hours=8)).strftime('%Y%m%d')
                input_file = ' '.join(
                    [glob.glob(os.path.join(self.path_cfg['input_dir'], time_str, str(file_id), "*"))[0]
                     for file_id in input_file_id])
            else:
                time_str = re.search(r'\d{14}', os.path.basename(file_info[0].get('filePath'))).group()  # 源文件为世界时
                time_str = (datetime.strptime(time_str, '%Y%m%d%H%M%S') - timedelta(minutes=1)).strftime('%Y%m%d')
                t_str = re.search(r'\d{14}', os.path.basename(file_info[0].get('filePath'))).group()
                t_str = (datetime.strptime(t_str, '%Y%m%d%H%M%S') - timedelta(minutes=1)).strftime('%Y%m%d%H%M%S')
                temp_path = [os.path.join(self.path_cfg['input_dir'], time_str, str(fi.get('fileId')),
                                          f'Z_RADR_I_{fi.get("equipNum")}_{t_str}_O_DOR_ETW_CAP_FMT.nc')
                             for fi in file_info]
                try:
                    input_file = ' '.join([file_exit(transfer_path(fi, is_win_path=True), sleep_times=2) for fi in temp_path])
                except NameError:
                    input_file = ' '.join(
                        [file_exit(transfer_path(fi.replace('CAP', 'CRA'), is_win_path=True), sleep_times=2) for fi in temp_path])

            output_data_path = os.path.join(self.path_cfg["exe_out_path"], time_str, file_label)  # 输出路径
            # output_data_inline_path = transfer_path(output_data_path, is_win_path=True, is_inline_path=True)  # 远程挂载路径
            try:
                rad_type = '_'.join([str(file['equipNum']) for file in file_info])
                t_str = re.search(r'\d{14}', os.path.basename(file_info[0].get('filePath'))).group()
                t_str = (datetime.strptime(t_str, '%Y%m%d%H%M%S') - timedelta(minutes=1)).strftime('%Y%m%d%H%M%S')
                file_absolute_path = transfer_path(os.path.join(output_data_path, f'DDA_{rad_type}_{t_str}.nc'),
                                                   is_win_path=True)
                listfile.append(file_exit(file_absolute_path, sleep_times=1))
                logger.info("文件{}无需执行算法程序".format(listfile[-1]))
            except NameError:
                # if glob.glob(f"{output_data_inline_path}/DDA_*"):
                #     listfile.append(glob.glob(f"{output_data_inline_path}/DDA_*")[0])
                #     logger.info("文件{}无需执行算法程序".format(listfile[-1]))
                #     continue
                # else:
                start_time = time.time()
                cmd = f"{self.exe_path} {output_data_path} {input_file}"
                os.system(cmd)
                logger.info("DDAWIND耗时{}".format(time.time() - start_time))

                dda_file = glob.glob(f"{output_data_path}/DDA_*")
                if not dda_file:
                    raise Exception("程序执行出错，未生成文件")
                else:
                    #  ---中间文件生成消息发送---
                    temp_file_info = temp_message(dda_file, input_file_id)
                    publish_temp_msg("20", self.jobid, "DDAWind", temp_file_info)

                listfile.append(dda_file[0])

        return listfile

    def get_data(self, point_lon=None, point_lat=None):
        """
        得到绘图的数据：
        若没有单点经纬度数据，则返回对应时间的数据信息列表
        若输入所选取的经纬度点的位置，返回单点的风场数据，绘制单点时间高度图。
        ----------
        param: point_lon: 位置经度点
        param: point_lat: 位置纬度点
        Returns: 字典数据
        -------
        """

        draw_u = np.empty(shape=(12, len(self.listfile)))
        draw_v = np.empty(shape=(12, len(self.listfile)))
        draw_w = np.empty(shape=(12, len(self.listfile)))
        high = None
        Grids = []

        for ti, files in enumerate(self.listfile):

            Grids.append(read_grid(files))
            if not (point_lon and point_lat):
                continue
            high = Grids[ti].point_altitude['data']
            u = Grids[ti].fields['u']['data']
            v = Grids[ti].fields['v']['data']
            w = Grids[ti].fields['w']['data']
            # self.height_list = Grids[ti].z['data']
            index, indey = self.get_point_index(Grids[ti], point_lon, point_lat)
            draw_u[:, ti] = u[:, indey, index]
            draw_v[:, ti] = v[:, indey, index]
            draw_w[:, ti] = w[:, indey, index]
            high = high[:, 0, 0]
            draw_u[draw_u == -9999] = None
            draw_v[draw_v == -9999] = None
            draw_w[draw_w == -9999] = None

        data = {'data_u': draw_u,
                'data_v': draw_v,
                'data_w': draw_w,
                'high': high,
                'Grids': Grids}
        return data

    def get_point_index(self, grids, point_lon, point_lat):
        """
        得到所选坐标点的位置序列
        ----------
        param: grids: 文件数据
        param: point_lon: 绘图坐标点经度
        param: point_lat: 绘图坐标点纬度
        return: 坐标的位置序列
        ----------
        """

        lats = grids.point_latitude['data'][0]
        lons = grids.point_longitude['data'][0]
        x = abs(lons - point_lon)
        y = abs(lats - point_lat)
        index = np.where(x == np.min(x))[1]
        indey = np.where(y == np.min(y))[0]

        if len(index) > 1:
            index = index[1]
        else:
            index = index[0]
        if len(indey) > 1:
            indey = indey[1]
        else:
            indey = indey[0]

        return index, indey

    def draw_horizontal_wind(self, idx, data, sta_point=None, end_point=None):
        """
        绘制各层水平风场图
        ----------
        param: idx: 时间序列参数
        param: grids: 数据信息
        param: sta_point: 起始经纬度坐标点：[经度，纬度]
        param: end_point: 结束经纬度坐标点：[经度，纬度]
        return: 图片信息
        ----------
        """

        fig = plt.figure(figsize=(6, 5))
        Grids = data

        barb_spacing_x_km = 3
        barb_spacing_y_km = 3
        barb_len = 2.5
        barb_line_width = 0.2
        if sta_point and end_point:
            min_lon = min(sta_point[0], end_point[0])
            max_lon = max(sta_point[0], end_point[0])
            min_lat = min(sta_point[1], end_point[1])
            max_lat = max(sta_point[1], end_point[1])
            barb_spacing_x_km = ((max_lon - min_lon) * 111) / 15
            barb_spacing_y_km = ((max_lat - min_lat) * 110) / 15
            barb_len = 4.5
            barb_line_width = 0.5
        background_field = 'reflectivity'

        grid_array = np.ma.stack(Grids.fields[background_field]['data'])
        grid_x = Grids.point_x['data'] / 1e3
        grid_y = Grids.point_y['data'] / 1e3
        grid_lat = Grids.point_latitude['data'][self.level]
        grid_lon = Grids.point_longitude['data'][self.level]
        u = Grids.fields['u']['data']
        v = Grids.fields['v']['data']

        # ax = fig.add_subplots()
        s_point = [103.01042080, 25.960774]
        d_point = [104.836801443, 27.58676280]
        my_map = Basemap(projection='merc', llcrnrlat=s_point[1], urcrnrlat=d_point[1],
                         llcrnrlon=s_point[0], urcrnrlon=d_point[0])
        x, y = my_map(grid_lon, grid_lat)

        dx = np.diff(grid_x, axis=2)[0, 0, 0]
        dy = np.diff(grid_y, axis=1)[0, 0, 0]
        barb_density_x = int((1 / dx) * barb_spacing_x_km) or 1
        barb_density_y = int((1 / dy) * barb_spacing_y_km) or 1

        cmap = ListedColormap(colormap['ref']["colors"])
        norm = BoundaryNorm(boundaries=colormap['ref']["levels"], ncolors=14)
        if self.threshold is not None:
            try:
                selected_data = select_data(np.array(grid_array[self.level, :, :]), 'ref', self.threshold)
            except ValueError:
                raise Exception(f"threshold参数错误{self.threshold}")
        else:
            selected_data = grid_array[self.level, :, :]
        mesh = my_map.pcolormesh(x[:, :], y[:, :], selected_data,
                                 cmap=cmap, norm=norm, zorder=0, shading='auto')
        my_map.barbs(x[::barb_density_y, ::barb_density_x],
                     y[::barb_density_y, ::barb_density_x],
                     u[self.level, ::barb_density_y, ::barb_density_x],
                     v[self.level, ::barb_density_y, ::barb_density_x],
                     length=barb_len, linewidth=barb_line_width,
                     sizes={"spacing": 0.2, 'height': 0.5, 'width': 0.5, 'emptybarb': 0}, zorder=1,
                     barb_increments=dict(half=2, full=4, flag=20))

        plt.axis('off')
        transparent, pad_inches = True, 0
        pic_path = os.path.join(self.pic_file_path, f"uv_{self.time_index[idx]}.png")
        pic_info = {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                    "element": "uv",
                    "otherData": {
                        "gisPicLonLat": [s_point, d_point],
                        "args": [{
                            "argsType": "height", "defaultValue": float(data.z['data'][0]),
                            "specialArgs": data.z['data'].tolist(), "unit": "m"}
                        ]}
                    }

        if sta_point and end_point:
            high = Grids.point_altitude['data'][:, 0, 0]
            high = high - high[0] + 1000
            x1, y1 = my_map(sta_point[0], sta_point[1])
            x2, y2 = my_map(end_point[0], end_point[1])
            sta_lon, end_lon = min(x1, x2), max(x1, x2)
            sta_lat, end_lat = min(y1, y2), max(y1, y2)
            plt.xlim(sta_lon, end_lon)
            plt.ylim(sta_lat, end_lat)
            plt.axis('on')
            lons = np.arange(sta_lon, end_lon + 1, (end_lon - sta_lon) / 4)
            lats = np.arange(sta_lat, end_lat + 1, (end_lat - sta_lat) / 4)
            x_lons, y_lats = my_map(lons, lats, inverse=True)
            plt.xticks(ticks=lons, labels=[f'{round(x_lons[i], 2)}°E' for i in range(5)])
            fig.autofmt_xdate()
            plt.yticks(ticks=lats, labels=[f'{round(y_lats[i], 2)}°N' for i in range(5)])
            plt.xlabel('经度')
            plt.ylabel('纬度')
            time_title = datetime.strptime(self.time_index[idx], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M')
            plt.title(f'水平风场图  高度：{high[self.level] / 1000}km {time_title}')
            br = plt.colorbar(mesh, fraction=0.023, pad=0.02, ticks=colormap['ref']["levels"],
                              aspect=30)  # 依次调整colorbar大小，与子图距离，label，横纵比
            br.ax.set_title('dBZ', fontsize=8)
            br.ax.tick_params()
            transparent, pad_inches = False, 0.1
            pic_path = os.path.join(self.pic_file_path, f"uv_{self.time_index[idx]}.png")
            pic_info = {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                        "element": "uv",
                        "otherData": {
                            "args": [{
                                "argsType": "height", "defaultValue": float(data.z['data'][0]),
                                "specialArgs": data.z['data'].tolist(), "unit": "m"}
                            ]}
                        }
        plt.savefig(pic_path, dpi=300, transparent=transparent, bbox_inches="tight", pad_inches=pad_inches)
        plt.close()

        # 上传图片
        # # minio_client.put_object(pic_path)
        return pic_info

    def draw_profile_wind(self, idx, grids, sta_point, end_point, blast_height=0, ele='w'):
        """
        两点间的剖面图
        ----------
        param: idx: 时间序列参数
        param: grids: 数据信息
        param: start: 起始点经纬度坐标点 [经度，纬度]
        param: end: 结束点经纬度坐标点 [经度，纬度]
        return: 图片信息
        ----------
        """

        start = sta_point[::-1]
        end = end_point[::-1]
        lat = grids.point_latitude['data'][0, :, 0]
        lon = grids.point_longitude['data'][0, 0, :]
        height = grids.point_altitude['data'][:, 0, 0]
        height = (height - height[0] + 1000) / 1000  # 将海拔高度转为距离雷达高度 km
        ref = np.ma.stack(grids.fields['reflectivity']['data'])
        u = grids.fields['u']['data']
        v = grids.fields['v']['data']
        w = grids.fields['w']['data']

        section_data = xr.Dataset(data_vars={'u': (['height', 'lat', 'lon'], u),
                                             'v': (['height', 'lat', 'lon'], v),
                                             'w': (['height', 'lat', 'lon'], w),
                                             'ref': (['height', 'lat', 'lon'], ref)},
                                  coords={'lon': (['lon'], lon),
                                          'lat': (['lat'], lat),
                                          'height': (['height'], height)}
                                  )
        section_data = section_data.metpy.parse_cf().squeeze()

        extend_for = start
        extend_back = end
        if blast_height:
            # extend_for = extend_point(end[0], end[-1], start[0], start[-1])
            extend_back = extend_point(start[0], start[-1], end[0], end[-1])

        cross = cross_section(section_data, extend_for, extend_back)
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.rcParams['font.size'] = 12

        if 'uvw' in ele:
            x = cross['index']
            y = cross['height']

            # 计算风向
            wdir_vert = np.arctan2(cross.v, cross.u) * 180 / np.pi
            line_angel = np.arctan2(0, - abs(end_point[-1] - sta_point[-1])) * 180 / np.pi
            vl_angel = np.cos(np.deg2rad(wdir_vert - line_angel))
            ws_vert = np.sqrt(cross.u ** 2 + cross.v ** 2) * vl_angel  # 计算风速
            q_u = plt.quiver(x[::10], y, ws_vert[:, ::10], cross.w[:, ::10],
                             scale=150, width=0.004, units='width')
            ax.quiverkey(q_u, 0.9, 0.08, 10, angle=0, label='10m/s', labelpos='S',
                         coordinates='figure', fontproperties={'size': 8})
            title_temp = '合成风'
        else:
            x = cross['index']
            # 设置等高线合适的间距
            if w.min() > -1:
                levels_min = np.arange(-1, 0, 0.2)
            else:
                levels_min = np.arange(np.around(w.min(), 1), 0, np.around(abs(w.min() / 10), 1))
            if w.max() < 1:
                levels_max = np.arange(0, 1, 0.2)
            else:
                levels_max = np.arange(0, np.around(w.max(), 1), np.around(abs(w.max() / 10), 1))

            cc_po = plt.contour(x, cross['height'], cross['w'],
                                levels=levels_min, colors='k', linewidths=1.5, linestyles='--')
            cc_ne = plt.contour(x, cross['height'], cross['w'],
                                levels=levels_max, colors='k', linewidths=1.5, linestyles='-')
            if not np.isnan(cross['w'].data).all():
                plt.clabel(cc_po, inline=True)
                plt.clabel(cc_ne, inline=True)
            title_temp = '垂直风(m/s)'
            plt.text(x=0, y=12.5, s='———上升气流 ----下沉气流')
        cmap = ListedColormap(colormap['ref']["colors"])
        norm = BoundaryNorm(boundaries=colormap['ref']["levels"], ncolors=14)
        plt.pcolormesh(x, cross['height'], cross['ref'],
                       cmap=cmap, norm=norm, zorder=0, shading='auto')

        r = geodesic(start, end)  # 爆炸点间离
        # ----添加爆炸点---------------
        if blast_height:
            station = int((1 - (r / geodesic(extend_for, extend_back))) * 100)  # 爆炸点位置
            ax.scatter(x[-station], blast_height / 1000, s=50, c='none', marker="^", edgecolors='k', linewidths=1.5)
            plt.xlabel("距离(km)")
            r = geodesic(extend_for, extend_back)

        # plt.title(self.title)
        br = plt.colorbar(fraction=0.027, pad=0.02, ticks=colormap['ref']["levels"], aspect=30)
        br.ax.set_title('dBZ')

        inter = get_limit_index(len(x), num=6)
        ticks = cross['index'].data[::inter]
        ys = np.round(cross['lon'].data[::inter], 2)
        xs = np.round(cross['lat'].data[::inter], 2)
        labels = []

        r_lab = np.round(np.linspace(0, r.km, 100)[::inter], 2)
        time_str = datetime.strptime(self.time_index[idx], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M')
        for n, y_val in enumerate(ys):
            lo = f'{abs(y_val)}°E' if y_val > 0 else f'{abs(y_val)}°W'
            la = f'{xs[n]}°N' if xs[n] > 0 else f'{xs[n]}°S'
            if blast_height:
                labels.append(str(r_lab[n]))
                title = f'{time_str}{self.title} {title_temp}'
            else:
                labels.append(f'{lo}\n{la}')
                title = f'{time_str} {title_temp}'

        plt.title(title, fontsize=FONTSIZE + 3)
        plt.xticks(ticks, labels)

        plt.ylim([0, height[-1] + 1])
        plt.ylabel('高度(km)')
        pic_path = os.path.join(self.pic_file_path, f"{ele}_{self.time_index[idx]}.png")
        plt.savefig(pic_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        # 上传图片
        # # minio_client.put_object(pic_path)
        pic_info = {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                    "element": ele}

        return pic_info

    def draw_profile_wind_sum(self, sta_point, end_point, blast_height=0, elements=None, eff_time=None):
        """
        两点间的剖面图
        ----------
        param: idx: 时间序列参数
        param: grids: 数据信息
        param: start: 起始点经纬度坐标点 [经度，纬度]
        param: end: 结束点经纬度坐标点 [经度，纬度]
        return: 图片信息
        ----------
        """
        data = self.get_data()
        eles = [el.split('_')[0] for el in elements]

        eff_times = [datetime.strptime(eff_t, '%Y%m%d%H%M%S') for eff_t in eff_time[0]]
        plt.rcParams['font.size'] = 12
        nrows, ncols = len(eles), len(self.listfile)
        time_list = [datetime.strptime(t, '%Y%m%d%H%M%S') for t in self.time_index]

        figsize = (6 * ncols, 5 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=400, sharex="all", sharey="all")
        for ele_idx, ele in enumerate(eles):
            for time_idx, grids in enumerate(data["Grids"]):
                data_time = time_list[time_idx].strftime('%Y-%m-%d %H:%M')
                start = sta_point[::-1]
                end = end_point[::-1]
                lat = grids.point_latitude['data'][0, :, 0]
                lon = grids.point_longitude['data'][0, 0, :]
                height = grids.point_altitude['data'][:, 0, 0]
                height = (height - height[0] + 1000) / 1000  # 将海拔高度转为距离雷达高度 km
                ref = np.ma.stack(grids.fields['reflectivity']['data'])
                u = grids.fields['u']['data']
                v = grids.fields['v']['data']
                w = grids.fields['w']['data']

                section_data = xr.Dataset(data_vars={'u': (['height', 'lat', 'lon'], u),
                                                     'v': (['height', 'lat', 'lon'], v),
                                                     'w': (['height', 'lat', 'lon'], w),
                                                     'ref': (['height', 'lat', 'lon'], ref)},
                                          coords={'lon': (['lon'], lon),
                                                  'lat': (['lat'], lat),
                                                  'height': (['height'], height)}
                                          )
                section_data = section_data.metpy.parse_cf().squeeze()
                extend_for = start
                extend_back = end
                if blast_height:
                    extend_back = extend_point(start[0], start[-1], end[0], end[-1])
                cross = cross_section(section_data, extend_for, extend_back)

                # %%
                if ncols == 1 and nrows == 1:
                    ax = axs
                elif ncols == 1:
                    ax = axs[ele_idx]
                elif nrows == 1:
                    ax = axs[time_idx]
                else:
                    ax = axs[ele_idx, time_idx]

                if 'uvw' in ele:
                    x = cross['index']
                    y = cross['height']

                    # 计算风向
                    wdir_vert = np.arctan2(cross.v, cross.u) * 180 / np.pi
                    line_angel = np.arctan2(0, - abs(end_point[-1] - sta_point[-1])) * 180 / np.pi
                    vl_angel = np.cos(np.deg2rad(wdir_vert - line_angel))
                    ws_vert = np.sqrt(cross.u ** 2 + cross.v ** 2) * vl_angel  # 计算风速
                    q_u = ax.quiver(x[::10], y, ws_vert[:, ::10], cross.w[:, ::10],
                                     scale=150, width=0.004, units='width')
                    ax.quiverkey(q_u, 0.9, 0.08, 10, angle=0, label='10m/s', labelpos='S',
                                 coordinates='figure', fontproperties={'size': 8})
                    title_temp = '风场结构'
                else:
                    x = cross['index']
                    # 设置等高线合适的间距
                    if w.min() > -1:
                        levels_min = np.arange(-1, 0, 0.2)
                    else:
                        levels_min = np.arange(np.around(w.min(), 1), 0, np.around(abs(w.min() / 10), 1))
                    if w.max() < 1:
                        levels_max = np.arange(0, 1, 0.2)
                    else:
                        levels_max = np.arange(0, np.around(w.max(), 1), np.around(abs(w.max() / 10), 1))

                    cc_po = ax.contour(x, cross['height'], cross['w'],
                                       levels=levels_min, colors='k', linewidths=1.5, linestyles='--')
                    cc_ne = ax.contour(x, cross['height'], cross['w'],
                                       levels=levels_max, colors='k', linewidths=1.5, linestyles='-')
                    if not np.isnan(cross['w'].data).all():
                        plt.clabel(cc_po, inline=True)
                        plt.clabel(cc_ne, inline=True)
                    title_temp = '垂直风(m/s)'
                    ax.text(x=0, y=12.5, s='———上升气流 ----下沉气流')
                cmap = ListedColormap(colormap['ref']["colors"])
                norm = BoundaryNorm(boundaries=colormap['ref']["levels"], ncolors=14)
                im = ax.pcolormesh(x, cross['height'], cross['ref'],
                                   cmap=cmap, norm=norm, zorder=0, shading='auto')
                ax.set_ylim([0, height[-1] + 1])

                r = geodesic(start, end)  # 爆炸点间离
                # ----添加爆炸点---------------
                if blast_height:
                    station = int((1 - (r / geodesic(extend_for, extend_back))) * 100)  # 爆炸点位置
                    ax.scatter(x[-station], blast_height / 1000, s=50, c='none', marker="^", edgecolors='k', linewidths=1.5)
                    r = geodesic(extend_for, extend_back)

                if time_list[time_idx] > eff_times[-1]:
                    title = '作业后\t' + data_time
                elif time_list[time_idx] < eff_times[0]:
                    title = '作业前\t' + data_time
                else:
                    title = '作业中\t' + data_time
                if ele_idx == 0:
                    ax.set_title(title)
                if ele_idx == nrows - 1:
                    inter = get_limit_index(len(x), num=6)
                    ticks = cross['index'].data[::inter]
                    ys = np.round(cross['lon'].data[::inter], 2)
                    xs = np.round(cross['lat'].data[::inter], 2)
                    labels = []

                    r_lab = np.round(np.linspace(0, r.km, 100)[::inter], 2)
                    for n, y_val in enumerate(ys):
                        lo = f'{abs(y_val)}°E' if y_val > 0 else f'{abs(y_val)}°W'
                        la = f'{xs[n]}°N' if xs[n] > 0 else f'{xs[n]}°S'
                        if blast_height:
                            labels.append(str(r_lab[n]))
                            title = f'{data_time}{self.title} {title_temp}'
                        else:
                            labels.append(f'{lo}\n{la}')
                            title = f'{data_time} {title_temp}'
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels)
                    ax.set(xlabel='距离(km)')
                if time_idx == 0:
                    ax.set(ylabel='高度(km)')

            cax_0 = add_right_cax(ax, pad=0.03 / ncols, width=0.03 / ncols)
            cb = fig.colorbar(im, ax=ax, cax=cax_0, ticks=colormap['ref']["levels"])
            cb.ax.set_title('dBZ')
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        pic_path = os.path.join(self.pic_file_path, f"{eles[0]}_{self.time_index[-1]}.png")
        plt.savefig(pic_path, dpi=400, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        # 上传图片
        # # minio_client.put_object(pic_path)
        pic_info = {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                    "element": eles[0]}

        return pic_info

    def draw_part_profile(self, idx, grids, sta_point, end_point, mid_point):
        """
        区域沿经、纬度剖面图
        ----------
        param: idx: 时间序列参数
        param: grids: 数据信息
        param: sta_point: 起始经纬度坐标点：[经度，纬度]
        param: end_point: 结束经纬度坐标点：[经度，纬度]
        return: 图片信息
        ----------
        """

        grid_lat = grids.point_latitude['data']
        grid_lon = grids.point_longitude['data']
        height = grids.point_altitude['data']
        height = (height - height[0] + 1000) / 1000  # 将海拔高度转为距离雷达高度 km
        ref = np.ma.stack(grids.fields['reflectivity']['data'])
        u_r = grids.fields['u']['data']
        v_r = grids.fields['v']['data']
        w = grids.fields['w']['data']
        sta_lon, sta_lat = self.get_point_index(grids, sta_point[0], sta_point[1])
        end_lon, end_lat = self.get_point_index(grids, end_point[0], end_point[1])
        min_lat = min(sta_lat, end_lat)
        max_lat = max(sta_lat, end_lat)
        min_lon = min(sta_lon, end_lon)
        max_lon = max(sta_lon, end_lon)

        if mid_point:
            lon_index, lat_index = self.get_point_index(grids, mid_point[0], mid_point[1])
        else:
            lat_index = min_lat
            lon_index = min_lon

        u = u_r[:, lat_index, min_lon:max_lon]
        v_u = v_r[:, lat_index, min_lon:max_lon]
        w_u = w[:, lat_index, min_lon:max_lon]
        ref_u = ref[:, lat_index, min_lon:max_lon]
        lon = grid_lon[:, lat_index, min_lon:max_lon]
        height_u = height[:, lat_index, min_lon:max_lon]

        v = v_r[:, min_lat:max_lat, lon_index]
        u_v = u_r[:, min_lat:max_lat, lon_index]
        w_v = w[:, min_lat:max_lat, lon_index]
        ref_v = ref[:, min_lat:max_lat, lon_index]
        lat = grid_lat[:, min_lat:max_lat, lon_index]
        height_v = height[:, min_lat:max_lat, lon_index]
        # 设置风羽间隔
        barb_spacing_x_u = round(len(height_u[1]) / 10) or 1
        barb_spacing_x_v = round(len(height_v[1]) / 10) or 1

        cmap = ListedColormap(colormap['ref']["colors"])
        norm = BoundaryNorm(boundaries=colormap['ref']["levels"], ncolors=14)

        fig = plt.figure(figsize=(7, 3))
        ax = plt.subplot(1, 2, 1)
        mesh = plt.pcolormesh(lon, height_u, ref_u, cmap=cmap, norm=norm, zorder=0, shading='auto')
        # plt.barbs(lon[::1, ::barb_spacing_x_u], height_u[::1, ::barb_spacing_x_u],
        #           u[::1, ::barb_spacing_x_u], w_u[::1, ::barb_spacing_x_u],
        #           length=4, linewidth=0.5,
        #           sizes={"spacing": 0.2, 'height': 0.5, 'width': 0.5, 'emptybarb': 0}, zorder=0,
        #           barb_increments=dict(half=2, full=4, flag=20))

        # 计算风向
        wdir_vert = np.arctan2(v_u, u) * 180 / np.pi
        line_angel = np.arctan2(0, - abs(end_point[0] - sta_point[0])) * 180 / np.pi
        vl_angel = np.cos(np.deg2rad(wdir_vert - line_angel))
        ws_vert = np.sqrt(u ** 2 + v_u ** 2) * vl_angel  # 计算风速
        q_u = plt.quiver(lon[::1, ::barb_spacing_x_u], height_u[::1, ::barb_spacing_x_u],
                         ws_vert[::1, ::barb_spacing_x_u], w_u[::1, ::barb_spacing_x_u],
                         scale=150, width=0.004, units='width')
        ax.quiverkey(q_u, 0.445, 0.9, 10, angle=0, label='10m/s', labelpos='W',
                     coordinates='figure', fontproperties={'size': 5})
        # ax.quiverkey(q_u, 0.45, 0.92, 10, angle=90, label='w:10m/s', labelpos='N',
        #              coordinates='figure', fontproperties={'size': 5})
        step_lon = get_limit_index(lon.shape[1], 5)
        plt.xticks(ticks=lon[0][::step_lon], labels=[f'{round(lo, 2)}°E' for lo in lon[0][::step_lon]])
        fig.autofmt_xdate()
        plt.ylim([0, height_u[-1, -1] + 1])
        # plt.xlabel('经度')
        plt.ylabel('高度(km)')
        plt.title('纬向剖面')

        ax2 = plt.subplot(1, 2, 2)
        mesh = plt.pcolormesh(lat, height_v, ref_v, cmap=cmap, norm=norm, zorder=0, shading='auto')
        # plt.barbs(lat[::1, ::barb_spacing_x_v], height_v[::1, ::barb_spacing_x_v],
        #           w_v[::1, ::barb_spacing_x_v], v[::1, ::barb_spacing_x_v],
        #           length=4, linewidth=0.5,
        #           sizes={"spacing": 0.2, 'height': 0.5, 'width': 0.5, 'emptybarb': 0}, zorder=0,
        #           barb_increments=dict(half=2, full=4, flag=20))

        # 计算风向
        wdir_vert = np.arctan2(v, u_v) * 180 / np.pi
        line_angel = np.arctan2(- abs(end_point[1] - sta_point[1]), 0) * 180 / np.pi
        vl_angel = np.cos(np.deg2rad(wdir_vert - line_angel))
        ws_vert = np.sqrt(u_v ** 2 + v ** 2) * vl_angel  # 计算风速
        q_v = plt.quiver(lat[::1, ::barb_spacing_x_v], height_v[::1, ::barb_spacing_x_v],
                         ws_vert[::1, ::barb_spacing_x_v], w_v[::1, ::barb_spacing_x_v],
                         scale=150, width=0.004, units='width')
        ax2.quiverkey(q_v, 0.845, 0.9, 10, angle=0, label='10m/s', labelpos='W',
                      coordinates='figure', fontproperties={'size': 5})
        # ax2.quiverkey(q_v, 0.85, 0.92, 10, angle=90, label='w:10m/s', labelpos='N',
        #               coordinates='figure', fontproperties={'size': 5})

        br = plt.colorbar(mesh, fraction=0.03, pad=0.02, ticks=colormap['ref']["levels"], aspect=30)
        br.ax.set_title('dBZ', fontsize=8)
        step_lat = get_limit_index(lat.shape[1], 5)
        plt.xticks(ticks=lat[0][::step_lat], labels=[f'{round(lo, 2)}°N' for lo in lat[0][::step_lat]])
        fig.autofmt_xdate()
        plt.ylim([0, height_v[-1, -1] + 1])
        # plt.xlabel('纬度')
        # plt.ylabel('高度(km)')
        plt.title('经向剖面')
        time_title = datetime.strptime(self.time_index[idx], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M')
        if self.title is None:
            self.title = ''
        plt.suptitle(f'{self.title} {time_title}', fontsize=10, y=1)

        pic_path = os.path.join(self.pic_file_path, f"w_{self.time_index[idx]}.png")
        plt.savefig(pic_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # # minio_client.put_object(pic_path)
        pic_info = {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                    "element": "w"}

        return pic_info

    def draw_point_time_high(self, types, data, eff_time, hail_time=None):
        """
        单点水平气流风羽及垂直气流填色图
        ----------
        param: types: 绘制图的类别 [uv:水平风，w:垂直气流]
        param: data: 绘图数据
        Returns: 图片信息
        -------
        """

        draw_u = data['data_u']
        draw_v = data['data_v']
        draw_w = data['data_w']
        if np.isnan(draw_u).all():
            draw_u = np.zeros_like(draw_u)
            draw_v = np.zeros_like(draw_u)
            draw_w = np.zeros_like(draw_u)
        high = data['high']
        high = (high - high[0] + 1000) / 1000

        fig, ax = plt.subplots(figsize=(12, 5))
        sta_ti = datetime.strptime(self.time_index[0], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M')
        end_ti = datetime.strptime(self.time_index[-1], '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M')
        t = [datetime.strptime(ti, '%Y%m%d%H%M%S') for ti in self.time_index]
        # ax.set_xlim(t[0], t[-1])
        kwargs_major, kwargs_minor, timestyle = set_x_time(t)
        if types == 'uv':
            # t = np.arange(0, len(self.time_index), 1)
            t, h = np.meshgrid(t, high)
            plt.barbs(t, h, draw_u, draw_v,
                      length=5, linewidth=0.5, sizes={"spacing": 0.2, 'height': 0.5, 'width': 0.3, 'emptybarb': 0},
                      barb_increments=dict(half=2, full=4, flag=20))
            plt.title(f'水平风(风羽图) {sta_ti}~{end_ti}', fontsize=FONTSIZE + 8)

        if types == 'w':
            levels = [-10, -8, -6, -5, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]
            norm = BoundaryNorm(boundaries=levels, ncolors=len(levels))
            listc = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]
            cmap = LinearSegmentedColormap.from_list('Rd_Bl_R', listc, N=len(levels))
            if len(t) == 1:
                raise ValueError('该时段数据量缺失不足以绘制填色图')
            plt.pcolormesh(t, high, draw_w,
                           cmap=cmap, norm=norm, shading='auto')
            plt.title(f'垂直风 {sta_ti}~{end_ti}', fontsize=FONTSIZE + 8)
            # tick = np.arange(-12, 12.1, 1.2)
            cb = plt.colorbar(fraction=0.027, pad=0.02, aspect=30, ticks=levels)
            cb.ax.set_title('m/s', fontsize=FONTSIZE)
            cb.ax.tick_params(labelsize=FONTSIZE)

        ax_add_time_range(ax, eff_time, high, alpha=0.3, color='red')
        ax_add_time_range(ax, hail_time, high, alpha=0.4, color='k')

        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)

        high = np.insert(high, [0], [0], axis=0)
        plt.xticks(fontsize=FONTSIZE + 2)
        plt.yticks(high, fontsize=FONTSIZE + 2)
        plt.xlabel('时间', fontsize=FONTSIZE + 2)
        plt.ylabel('高度(km)', fontsize=FONTSIZE + 2)
        pic_path = os.path.join(self.pic_file_path, f'{types}_{self.time_index[0]}.png')
        plt.savefig(pic_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # # minio_client.put_object(pic_path)
        pic_info = {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                    "element": types}

        return pic_info

    def run(self, types=None, point_lon=None, point_lat=None, sta_point=None, end_point=None, mid_point=None,
            is_part=None, blast_point=None, work_point=None, eff_time=None, blast_height=None, hail_time=None,
            eles=None):
        """
        ----------
        param: types: 绘图类型，uv:水平风，w:垂直气流
        param: point_lon: 单点经度
        param: point_lat: 单点纬度
        param: sta_point: 起始经纬度坐标点：[经度，纬度]
        param: end_point: 结束经纬度坐标点：[经度，纬度]
        param: ispart: 是否需要绘制区域
        param: blast_point: 爆炸点经纬度坐标点：[经度，纬度]
        param: work_point: 炮点经纬度坐标点：[经度，纬度]
        param: eff_time: 作业时间[[,], [,]]
        return: 生成的图片信息列表
        ----------
        """
        pic_file = []
        if not Path(self.pic_file_path).exists():
            Path.mkdir(Path(self.pic_file_path), parents=True, exist_ok=True)

        if eles:
            result = self.draw_profile_wind_sum(work_point, blast_point, blast_height, eles, eff_time)
            pic_file.append(result)
            # executor = ThreadPoolExecutor(max_workers=5)
            # executor.submit(upload_files, self.listfile)
            return [{"picFiles": pic_file}]
        if types is None:
            types = ['w', 'uvw']
        if work_point and blast_point:
            sta_point, end_point = work_point, blast_point
        elif not work_point and blast_point:
            point_lon, point_lat = blast_point

        if types[0] not in 'uvw':
            raise Exception("请说明绘制类型，如：'uv','w'")
        if point_lat and point_lon:
            point_data = self.get_data(point_lon, point_lat)
            result = self.draw_point_time_high(types, point_data, eff_time, hail_time=hail_time)
            pic_file.append(result)
        elif is_part:
            data = self.get_data()
            for idx, grid in enumerate(data["Grids"]):
                if types == 'uv':
                    result = self.draw_horizontal_wind(idx, grid, sta_point, end_point)
                    pic_file.append(result)
                elif types == 'w':
                    result = self.draw_part_profile(idx, grid, sta_point, end_point, mid_point)
                    pic_file.append(result)
        else:
            data = self.get_data()
            for idx, grid in enumerate(data["Grids"]):
                if types == 'uv':
                    result = self.draw_horizontal_wind(idx, grid)
                    pic_file.append(result)
                else:
                    if type(types) is str:
                        result = self.draw_profile_wind(idx, grid, sta_point, end_point, blast_height, types)
                        pic_file.append(result)
                    else:
                        for ele in types:
                            result = self.draw_profile_wind(idx, grid, sta_point, end_point, blast_height, ele)
                            pic_file.append(result)
        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, self.listfile)
        return [{"picFiles": pic_file}]
