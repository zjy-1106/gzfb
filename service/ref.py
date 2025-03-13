import os
import glob
import re
import itertools
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from config.config import config_info
from service.colormap import colormap
from service.dsd_rainrate_fusion import DsdRainRateController, fmt
from service.utils import transfer_path, file_exit, get_file_path, add_right_cax
# from utils.file_uploader import minio_client

mpl.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
FONTSIZE = 12


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
    # index_x = np.unravel_index(np.argmin(np.abs(lons - point_lon), axis=None), lons.shape)
    # index_y = np.unravel_index(np.argmin(np.abs(lats - point_lat), axis=None), lats.shape)
    return index, indey


class Reflectivity:
    def __init__(self, fileinfo, jobid, element, title, analysis_type, level=3000, elements=None):
        """
        ----------
        param: fileinfo: 传入的文件信息
        param: level: 绘制水平风场的层高
        param: jobid:
        param: element: 要素名
        param: title:
        param: analysis_type: 绘图分析类型
        ----------
        """

        self.fileinfo = fileinfo[0]
        self.title = title
        self.jobid = jobid
        self.analysis_type = analysis_type
        self.level = level if level else 3000
        self.height = []
        file_path = config_info.get_ref_cfg['input_dir']
        pic_path = config_info.get_ref_cfg['pic_path']
        self.eles = []
        if elements:
            self.eles = [el.split('_')[0] for el in elements]
            self.type = elements[0].split('_')[-1]
            self.nc_files = []
            if 'dm' in elements or 'nw' in elements:
                if len(fileinfo) == 1:
                    fileinfo = [fileinfo]
                for info in fileinfo:
                    self.nc_files.append(self.get_dsd_radar(info))
            self.nc_files = get_file_path(fileinfo, file_path, img_type=None)
        else:
            self.element = element
            self.ele = element.split('_')[0]
            self.type = element.split('_')[-1]
            for file_info in self.fileinfo:
                if file_info.get('radType') == 'HTM':
                    self.time_str = re.search(r'\d{14}',
                                              os.path.basename(file_info.get('filePath')).replace('_', '')).group()
                    time_day = (datetime.strptime(self.time_str, '%Y%m%d%H%M%S') - timedelta(hours=8)).strftime(
                        '%Y%m%d')
                else:
                    self.time_str = re.search(r'\d{14}', os.path.basename(file_info.get('filePath'))).group()  # 源文件为世界时
                    time_day = (datetime.strptime(self.time_str, '%Y%m%d%H%M%S')).strftime('%Y%m%d')
                break
            input_file_id = [str(f.get('fileId')) for f in self.fileinfo]
            inp = list(itertools.product(input_file_id, repeat=len(input_file_id)))
            for id_list in inp:
                try:
                    if 'dm' in self.element or 'nw' in self.element:
                        self.data_file = self.get_dsd_radar()
                    else:
                        self.data_file = ''.join(
                            glob.glob(os.path.join(file_path, time_day, '-'.join(id_list), "*"))[0])  # 文件路径
                    fi = True
                except IndexError as e:
                    fi = False
                    continue
                else:
                    break
            if not fi:
                raise Exception(f'文件未查询成功，请稍后重试。')

        self.pic_file_path = os.path.join(pic_path, str(self.jobid))  # 生成图片保存的路径

    def get_dsd_radar(self, file_info=None):
        if file_info:
            fileinfo = file_info
        else:
            fileinfo = self.fileinfo

        for file in fileinfo:
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
            path_cfg = config_info.get_dsd_rainrate_config
            temp_path = os.path.join(path_cfg['input_dir'], time_str, str(file_id),
                                     f'Z_RADR_I_{file.get("equipNum")}_{t_str}_O_DOR_ETW_CAP_FMT.nc')
            nc_file = file_exit(transfer_path(temp_path, is_win_path=True))
            file['filePath'] = nc_file

            return DsdRainRateController(file, self.element, 0, self.jobid).run_exe([file], t_str)[0]

    def get_data(self, point_lat=None, point_lon=None, str_point=None, end_point=None, mid_point=None):

        pic_data = []
        point_data = []
        with Dataset(self.data_file, 'r') as ds:
            # scw读取hcl
            ele = 'hcl' if self.ele == 'scw' else self.ele
            data = np.array(ds.variables[ele][:].data, dtype=np.float32)
            data[data == -999] = None
            lon = np.array(ds.variables["Dim3"][:].data, dtype=np.float32)
            lat = np.array(ds.variables["Dim2"][:].data, dtype=np.float32)
            self.height = np.array(ds.variables["Dim1"][:].data, dtype=int)

            temp1 = np.array([point_lat, point_lon], dtype=np.float32)
            temp2 = np.array([str_point, end_point], dtype=np.float32)
            if np.isnan(temp1).all() and np.isnan(temp2).all():
                pic_data = {
                    "data": data,
                    "lon": lon,
                    "lat": lat,
                    "height": self.height / 1000
                }
                return pic_data, point_data

            temp = np.array([point_lat, point_lon], dtype=np.float32)
            if not np.isnan(temp).all():
                lon_index, lat_index = get_point_index(lon, lat, point_lon, point_lat)
                if not lat_index or not lon_index:
                    raise Exception("该点（{},{}）不在目标范围内".format(np.round(point_lat, 2), np.round(point_lon, 2)))
                point_data = {
                    "data": data[:, lat_index, lon_index],
                    "height": self.height / 1000
                }
                return pic_data, point_data

            temp = np.array([str_point, end_point], dtype=np.float32)
            if not np.isnan(temp).all():
                lon_str, lat_str = get_point_index(lon, lat, str_point[0], str_point[1])
                lon_end, lat_end = get_point_index(lon, lat, end_point[0], end_point[1])
                lon_index_str, lon_index_end = min(lon_str, lon_end), max(lon_str, lon_end)
                lat_index_str, lat_index_end = min(lat_str, lat_end), max(lat_str, lat_end)
                if not lon_index_end or not lat_index_end or not lon_index_str or not lat_index_str:
                    raise Exception(
                        "该点（{},{}）（{},{}）不在目标范围内".format(np.round(str_point[0], 2), np.round(str_point[-1], 2),
                                                         np.round(end_point[0], 2), np.round(end_point[-1], 2)))
                lon_index_str = lon_index_str - 1 if lon_index_str == lon_index_end else lon_index_str
                lat_index_str = lat_index_str - 1 if lat_index_str == lat_index_end else lat_index_str
                # 扩展区域
                lon_index_str = lon_index_str - 10 if lon_index_str > 10 else 0
                lat_index_str = lat_index_str - 10 if lat_index_str > 10 else 0
                lon_index_end = lon_index_end + 10 if lon_index_end + 10 < len(lon) else len(lon) - 1
                lat_index_end = lat_index_end + 10 if lat_index_end + 10 < len(lon) else len(lon) - 1

            lon_index_mid, lat_index_mid = lon_index_str, lat_index_str
            if mid_point:
                lon_index_mid, lat_index_mid = get_point_index(lon, lat, mid_point[0], mid_point[1])

            pic_data = {
                "data": data[:, lat_index_str:lat_index_end + 1, lon_index_str:lon_index_end + 1],
                "lon": lon[lon_index_str:lon_index_end + 1],
                "lat": lat[lat_index_str:lat_index_end + 1],
                "mid_index": [lat_index_mid - lat_index_str, lon_index_mid - lon_index_str],
                "height": self.height / 1000
            }

        return pic_data, point_data

    def get_data_area(self, datafile, point_lat=None, point_lon=None, str_point=None, end_point=None, mid_point=None):
        with Dataset(datafile, 'r') as ds:
            # scw读取hcl
            lon = np.array(ds.variables["Dim3"][:].data, dtype=np.float32)
            lat = np.array(ds.variables["Dim2"][:].data, dtype=np.float32)
            self.height = np.array(ds.variables["Dim1"][:].data, dtype=int)

            temp1 = np.array([point_lat, point_lon], dtype=np.float32)
            temp2 = np.array([str_point, end_point], dtype=np.float32)
            if np.isnan(temp1).all() and np.isnan(temp2).all():
                return [0, -1], [0, -1]

            temp = np.array([str_point, end_point], dtype=np.float32)
            if not np.isnan(temp).all():
                lon_str, lat_str = get_point_index(lon, lat, str_point[0], str_point[1])
                lon_end, lat_end = get_point_index(lon, lat, end_point[0], end_point[1])
                lon_index_str, lon_index_end = min(lon_str, lon_end), max(lon_str, lon_end)
                lat_index_str, lat_index_end = min(lat_str, lat_end), max(lat_str, lat_end)
                if not lon_index_end or not lat_index_end or not lon_index_str or not lat_index_str:
                    raise Exception(
                        "该点（{},{}）（{},{}）不在目标范围内".format(np.round(str_point[0], 2), np.round(str_point[-1], 2),
                                                         np.round(end_point[0], 2), np.round(end_point[-1], 2)))
                lon_index_str = lon_index_str - 1 if lon_index_str == lon_index_end else lon_index_str
                lat_index_str = lat_index_str - 1 if lat_index_str == lat_index_end else lat_index_str
                # 扩展区域
                lon_index_str = lon_index_str - 10 if lon_index_str > 10 else 0
                lat_index_str = lat_index_str - 10 if lat_index_str > 10 else 0
                lon_index_end = lon_index_end + 10 if lon_index_end + 10 < len(lon) else len(lon) - 1
                lat_index_end = lat_index_end + 10 if lat_index_end + 10 < len(lon) else len(lon) - 1

            lon_index_mid, lat_index_mid = lon_index_str, lat_index_str
            if mid_point:
                lon_index_mid, lat_index_mid = get_point_index(lon, lat, mid_point[0], mid_point[1])

        return [lon_index_str, lon_index_end + 1], [lat_index_str, lat_index_end + 1]

    def draw_hor(self, draw_data, time_s, mid_point=None):

        file_time = time_s.strftime('%Y%m%d%H%M')
        lons, lats = np.meshgrid(draw_data.get('lon'), draw_data.get('lat'))
        if lons.size == 1 or lats.size == 1:
            raise ValueError('区域过小，分辨率不足，无法绘制。')

        fig, ax = plt.subplots()  # figsize=(7, 5)
        plt.rcParams['font.size'] = FONTSIZE
        cmap = ListedColormap(colormap[self.ele]["colors"])
        norm = BoundaryNorm(boundaries=colormap[self.ele]["levels"], ncolors=len(colormap[self.ele]["levels"]))
        # level = int(self.level / 100 - 1)
        level = np.argmin(abs(np.array(self.height) - self.level)) if self.ele not in ['crf', 'etp', 'vil'] else 0
        pic = plt.pcolormesh(lons, lats, draw_data.get('data')[level, :, :],
                             norm=norm, cmap=cmap, shading='auto')

        x_lons = np.arange(lons[0, 0], lons[0, -1] + 0.0001, (lons[0, -1] - lons[0, 0]) / 4)
        y_lats = np.arange(lats[0, 0], lats[-1, 0] + 0.0001, (lats[-1, 0] - lats[0, 0]) / 4)
        plt.xticks(ticks=x_lons, labels=[f'{round(x_lons[i], 2)}°E' for i in range(5)])
        fig.autofmt_xdate()
        plt.yticks(ticks=y_lats, labels=[f'{round(y_lats[i], 2)}°N' for i in range(5)])
        if self.ele in ['crf', 'etp', 'vil']:
            plt.title(f'{self.title} {time_s.strftime("%Y-%m-%d %H:%M")}', fontsize=FONTSIZE)
        else:
            plt.title(f'{self.title} {time_s.strftime("%Y-%m-%d %H:%M")} 海拔高度：{draw_data.get("height")[level]}km',
                      fontsize=FONTSIZE)
        # br = plt.colorbar(pic, fraction=0.023, pad=0.02, ticks=colormap[self.ele]["levels"],
        #                   aspect=30)  # 依次调整colorbar大小，与子图距离，label，横纵比
        # br.ax.set_title('dBZ')
        if self.ele == 'scw':
            bins = [1, 2, 3]
            nbin = len(bins) - 1
            cmap = ListedColormap(['#a0a0a0', '#ff0000'])
            norm4 = mpl.colors.BoundaryNorm(bins, nbin)
            im4 = mpl.cm.ScalarMappable(norm=norm4, cmap=cmap)
            cb = fig.colorbar(im4, ax=ax)
        elif self.ele == 'nw':
            cb = fig.colorbar(pic, ticks=colormap[self.ele].get('levels'), shrink=0.9, pad=0.075,
                              format=mpl.ticker.FuncFormatter(fmt))
            cb.ax.set_title(colormap[self.ele]['label'], fontsize=10)
        else:
            cb = fig.colorbar(pic, ticks=colormap[self.ele].get('levels'), shrink=0.9, pad=0.035)
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

        if mid_point:
            # 质心点位置
            ax.scatter(mid_point[0], mid_point[1], s=50, c='none', marker="^", edgecolors='k', linewidths=1.5)
        plt.tick_params(labelsize=FONTSIZE)
        if not Path(self.pic_file_path).exists():
            Path.mkdir(Path(self.pic_file_path), parents=True, exist_ok=True)
        png_path = os.path.join(self.pic_file_path, f"hor_{file_time}_{self.ele}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": self.element,
                    "otherData": {
                        "args": [{
                            "argsType": "height", "defaultValue": float(3000), "argsDesc": "海拔高度",
                            "specialArgs": (draw_data.get("height") * 1000).tolist(), "unit": "m"}
                        ]}
                    }
        return pic_info

    def draw_hor_sum(self, str_point=None, end_point=None, mid_point=None, eff_time=None):
        eff_times = [datetime.strptime(eff_t, '%Y%m%d%H%M%S') for eff_t in eff_time[0]]
        plt.rcParams['font.size'] = FONTSIZE
        plt.tick_params(labelsize=FONTSIZE)
        nrows, ncols = len(self.eles), len(self.nc_files)
        lon_idx, lat_idx = self.get_data_area(self.nc_files[0], str_point=str_point, end_point=end_point)

        figsize = (6 * ncols, 5 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=400, sharex="all", sharey="all")
        for ele_idx, ele in enumerate(self.eles):
            for time_idx, nc_file in enumerate(self.nc_files):
                with Dataset(nc_file) as ds:
                    height = np.array(ds.variables["Dim1"][:].data, dtype=int)
                    level = np.argmin(abs(np.array(height) - self.level)) if ele not in ['crf', 'etp', 'vil'] else 0
                    if ele == 'scw':
                        data = np.array(ds.variables['hcl'][:].data, dtype=np.float32)
                    else:
                        data = np.array(ds.variables[ele][:].data, dtype=np.float32)
                    data[data == -999] = None
                    lon = np.array(ds.variables["Dim3"][:].data, dtype=np.float32)
                    lat = np.array(ds.variables["Dim2"][:].data, dtype=np.float32)
                    datas = data[level, lat_idx[0]:lat_idx[-1], lon_idx[0]:lon_idx[-1]]

                    if ncols == 1 and nrows == 1:
                        ax = axs
                    elif ncols == 1:
                        ax = axs[ele_idx]
                    elif nrows == 1:
                        ax = axs[time_idx]
                    else:
                        ax = axs[ele_idx, time_idx]

                    time_str = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y%m%d%H%M%S')
                    data_time = datetime.strftime(datetime.fromtimestamp(ds.getncattr('RadTime')), '%Y-%m-%d %H:%M')
                    lons, lats = np.meshgrid(lon[lon_idx[0]:lon_idx[-1]], lat[lat_idx[0]:lat_idx[-1]])
                    if lons.size == 1 or lats.size == 1:
                        raise ValueError('区域过小，分辨率不足，无法绘制。')

                    cmap = ListedColormap(colormap[ele]["colors"])
                    norm = BoundaryNorm(boundaries=colormap[ele]["levels"], ncolors=len(colormap[ele]["levels"]))
                    pic = ax.pcolormesh(lons, lats, datas, norm=norm, cmap=cmap, shading='auto')
                    # 质心点位置
                    if mid_point:
                        ax.scatter(mid_point[0], mid_point[1], s=50, c='none', marker="^", edgecolors='k',
                                   linewidths=1.5)

                    if datetime.fromtimestamp(ds.getncattr('RadTime')) > eff_times[-1]:
                        title = '作业后\t' + data_time
                    elif datetime.fromtimestamp(ds.getncattr('RadTime')) < eff_times[0]:
                        title = '作业前\t' + data_time
                    else:
                        title = '作业中\t' + data_time
                    if ele_idx == 0:
                        ax.set_title(title)
                    if ele_idx == nrows - 1:
                        x_lons = np.arange(lons[0, 0], lons[0, -1] + 0.0001, (lons[0, -1] - lons[0, 0]) / 4)
                        x_labels = [f'{round(x_lons[i], 2)}°E' for i in range(5)]
                        ax.set_xticks(x_lons)
                        ax.set_xticklabels(x_labels)
                    if time_idx == 0:
                        y_lats = np.arange(lats[0, 0], lats[-1, 0] + 0.0001, (lats[-1, 0] - lats[0, 0]) / 4)
                        y_labels = [f'{round(y_lats[i], 2)}°E' for i in range(5)]
                        ax.set_yticks(y_lats)
                        ax.set_yticklabels(y_labels)
            cax_0 = add_right_cax(ax, pad=0.03 / ncols, width=0.03 / ncols)
            if ele == 'scw':
                bins = [1, 2, 3]
                nbin = len(bins) - 1
                cmap = ListedColormap(['#a0a0a0', '#ff0000'])
                norm4 = mpl.colors.BoundaryNorm(bins, nbin)
                im4 = mpl.cm.ScalarMappable(norm=norm4, cmap=cmap)
                cb = fig.colorbar(im4, ax=ax, cax=cax_0)
            elif ele == 'nw':
                cb = fig.colorbar(pic, ax=ax, cax=cax_0, ticks=colormap[ele].get('levels'))
            else:
                cb = fig.colorbar(pic, ax=ax, cax=cax_0, ticks=colormap[ele].get('levels'))

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

        # plt.suptitle(f'{height[level]/1000}km', x=0.13, y=0.9)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        if not Path(self.pic_file_path).exists():
            Path.mkdir(Path(self.pic_file_path), parents=True, exist_ok=True)
        png_path = os.path.join(self.pic_file_path, f"hor_{time_str}_{self.eles[0]}.png")
        plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": self.eles[0],
                    "otherData": {
                        "args": [{
                            "argsType": "height", "defaultValue": float(3000), "argsDesc": "海拔高度",
                            "specialArgs": height.tolist(), "unit": "m"}
                        ]}
                    }
        return pic_info

    def draw_pro(self, draw_data, time_s):

        file_time = time_s.strftime('%Y%m%d%H%M')
        lon_h, h_lon = np.meshgrid(draw_data.get('lon'), draw_data.get('height'))
        lat_h, h_lat = np.meshgrid(draw_data.get('lat'), draw_data.get('height'))
        mid_index = draw_data.get('mid_index')

        fig = plt.figure(figsize=(7, 3))  # figsize=(7, 5)
        plt.subplot(1, 2, 1)
        cmap = ListedColormap(colormap[self.ele]["colors"])
        norm = BoundaryNorm(boundaries=colormap[self.ele]["levels"], ncolors=len(colormap[self.ele]["levels"]))
        pic = plt.pcolormesh(lon_h, h_lon, draw_data.get('data')[:, mid_index[0], :],
                             norm=norm, cmap=cmap, shading='auto')

        x_lons = np.arange(lon_h[0, 0], lon_h[0, -1] + 0.0001, (lon_h[0, -1] - lon_h[0, 0]) / 4)
        plt.xticks(ticks=x_lons, labels=[f'{round(x_lons[i], 2)}°E' for i in range(5)])
        # plt.yticks()
        fig.autofmt_xdate()
        plt.ylabel('高度(km)')
        plt.title('纬向剖面', fontsize=10)
        # br = plt.colorbar(pic, fraction=0.03, pad=0.02, ticks=colormap['ref']["levels"], aspect=30)
        # br.ax.set_title('dBZ')

        ax = plt.subplot(1, 2, 2)
        cmap = ListedColormap(colormap[self.ele]["colors"])
        norm = BoundaryNorm(boundaries=colormap[self.ele]["levels"], ncolors=len(colormap[self.ele]["levels"]))
        pic = plt.pcolormesh(lat_h, h_lat, draw_data.get('data')[:, :, mid_index[1]],
                             norm=norm, cmap=cmap, shading='auto')

        x_lats = np.arange(lat_h[0, 0], lat_h[0, -1] + 0.0001, (lat_h[0, -1] - lat_h[0, 0]) / 4)
        plt.xticks(ticks=x_lats, labels=[f'{round(x_lats[i], 2)}°N' for i in range(5)])
        fig.autofmt_xdate()
        # plt.ylabel('高度(km)')
        plt.title('经向剖面', fontsize=10)

        if self.ele == 'scw':
            bins = [1, 2, 3]
            nbin = len(bins) - 1
            cmap = ListedColormap(['#a0a0a0', '#ff0000'])
            norm4 = BoundaryNorm(bins, nbin)
            im4 = mpl.cm.ScalarMappable(norm=norm4, cmap=cmap)
            cb = fig.colorbar(im4, ax=ax)
        else:
            cb = plt.colorbar(pic, fraction=0.03, pad=0.035, ticks=colormap[self.ele]["levels"], aspect=30)
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
        # br = plt.colorbar(pic, fraction=0.03, pad=0.02, ticks=colormap[self.ele]["levels"], aspect=30)
        # br.ax.set_title('dBZ')

        plt.suptitle(f'{self.title} {time_s.strftime("%Y-%m-%d %H:%M")}', fontsize=11, y=1)

        if not Path(self.pic_file_path).exists():
            Path.mkdir(Path(self.pic_file_path), parents=True, exist_ok=True)
        png_path = os.path.join(self.pic_file_path, f"pro_{file_time}_{self.ele}.png")
        plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": self.element}
        return pic_info

    def run(self, point_lat=None, point_lon=None, str_point=None, end_point=None, mid_point=None, eff_time=None):

        # 转换北京时
        pic_info = []
        line_data = []

        if self.eles:
            if self.type == 'hor':  # 水平区域
                pic_info.append(self.draw_hor_sum(str_point=str_point, end_point=end_point, mid_point=mid_point,
                                                  eff_time=eff_time))
            else:
                raise ValueError("绘图失败，请输入正确参数：ref_hor")
        else:
            time_s = datetime.strptime(re.search(r'\d{14}', self.data_file).group(), '%Y%m%d%H%M%S') + timedelta(hours=8)
            logger.info("开始读取数据...")
            pic_data, point_data = self.get_data(point_lat, point_lon, str_point, end_point, mid_point)
            logger.info("开始绘图...")
            if self.analysis_type == 1:
                # 廓线数据
                temp = np.array([str_point, end_point], dtype=np.float32)
                x, x_max, x_mean, x_min = [], [], [], []
                if not np.isnan(temp).all():
                    area_data = pic_data.get('data')
                    area_data[area_data == -999] = None
                    for data_c in area_data:
                        x_max.append(float(np.nanmax(data_c)))
                        x_mean.append(float(np.nanmean(data_c)))
                        x_min.append(float(np.nanmin(data_c)))
                        y = np.around(pic_data.get('height'), decimals=1)
                else:
                    x = point_data.get('data').tolist()
                    y = np.around(point_data.get('height'), decimals=1)
                line_data = [{
                    "x": x,
                    "x_min": x_min,
                    "x_max": x_max,
                    "x_mean": x_mean,
                    "y": [y.tolist()],
                    "yname": [f'组网基本反射率垂直廓线 {str(time_s)}'],
                    "xlabel": ['dBZ'],
                    "ylabel": '高度(km)',
                    "time": str(time_s)
                }]
            elif self.analysis_type == 6:

                if self.type == 'hor':  # 水平区域
                    pic_info.append(self.draw_hor(pic_data, time_s, mid_point))
                elif self.type == 'pro':  # 区域垂直剖面
                    pic_info.append(self.draw_pro(pic_data, time_s))
                else:
                    raise ValueError("绘图失败，请输入正确参数：ref_pro, ref_hor")
            else:
                raise ValueError(f"analysis_type={self.analysis_type},请输入正确的绘图类型")

        return [{"picFiles": pic_info, "picData": [line_data]}]
