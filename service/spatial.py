import glob
import itertools
import os
import re
from datetime import datetime, timedelta

import plotly.graph_objects as go
import numpy as np
from pathlib import Path

from loguru import logger
from netCDF4 import Dataset

from config.config import config_info
from mq.pubulisher import publish_temp_msg
from service.colormap import colormap
from service.dda_wind import DDAWindController
from service.ref import Reflectivity
from service.utils import transfer_path
# from utils.file_uploader import minio_client

colo = colormap['ref']['colors']
co_l = np.linspace(0, 1, len(colo))
co_scale = []
for j in range(len(colo)):
    co_scale.append([co_l[j], colo[j]])
levels = colormap['ref']['levels']


class Spatial_3D:
    def __init__(self, fileinfo, element, jobid, level=15000):
        """
        ----------
        param: fileinfo: 传入的文件信息
        param: element: 产品标识
        param: jobid:
        param: level: 三维图的层高
        ----------
        """

        self.fileinfo = fileinfo
        self.ele = element
        self.jobid = jobid
        self.level = level if level else 15000
        file_path = config_info.get_spatial_cfg['input_dir_grid']
        picfile_path = config_info.get_spatial_cfg['pic_path']
        datafile_path = config_info.get_spatial_cfg['data_path']
        self.pic_path = os.path.join(picfile_path, str(self.jobid))
        self.datafile_path = os.path.join(datafile_path, str(self.jobid))
        if not Path(self.pic_path).exists():
            Path.mkdir(Path(self.pic_path), parents=True, exist_ok=True)
        if not Path(self.datafile_path).exists():
            Path.mkdir(Path(self.datafile_path), parents=True, exist_ok=True)

    def get_data(self, data_file, start_point=None, end_point=None):
        ds = Dataset(data_file)
        ele = self.ele.split('_')[0] if self.ele not in ['w', 'uv', '3D_structure'] else 'ref'
        if ele == 'scw':
            ele = 'hcl'  # 过冷水识别数据读取水凝物的
        data = np.array(ds.variables[ele][:])  # 需要绘制的数据
        flat = ds.variables['Dim2'][:]
        flon = ds.variables['Dim3'][:]
        layer = list(ds.variables['Dim1'][:])
        data[data == -999] = None

        if start_point and end_point:
            idy1 = np.argmin(np.abs(np.array(flon) - start_point[0]))
            idx1 = np.argmin(np.abs(np.array(flat) - start_point[-1]))
            idy2 = np.argmin(np.abs(np.array(flon) - end_point[0]))
            idx2 = np.argmin(np.abs(np.array(flat) - end_point[-1]))
            idx_s, idx_e = np.min([idx1, idx2]), np.max([idx1, idx2])
            idy_s, idy_e = np.min([idy1, idy2]), np.max([idy1, idy2])
            data = data[:, idx_s:idx_e, idy_s:idy_e]

        lats = np.linspace(flat[0], flat[-1], 5)
        lons = np.linspace(flon[0], flon[-1], 5)
        return data, flat, flon, layer, lats, lons

    def get_wind_data(self, data_file, start_point=None, end_point=None):
        ds = Dataset(data_file)
        ref = np.array(ds.variables['reflectivity'][:])[0]  # 需要绘制的数据
        # u = np.array(ds.variables['u'][:])[0]
        # v = np.array(ds.variables['v'][:])[0]
        # w = np.array(ds.variables['w'][:])[0]
        layer = np.array(ds.variables['point_altitude'][:])
        flat = np.array(ds.variables['point_latitude'][:])
        flon = np.array(ds.variables['point_longitude'][:])  # 数据层数
        layer = layer - ds.variables['origin_altitude'][:]
        lat_r = float(ds.variables['radar_latitude'][:])
        lon_r = float(ds.variables['radar_longitude'][:])
        lats = [lat_r - 0.2, lat_r - 0.1, lat_r, lat_r + 0.1, lat_r + 0.2]
        lons = [lon_r - 0.2, lon_r - 0.1, lon_r, lon_r + 0.1, lon_r + 0.2]
        ref[ref == -9999] = None
        if start_point and end_point:
            idy1 = np.argmin(np.abs(np.array(flon) - start_point[0]))
            idx1 = np.argmin(np.abs(np.array(flat) - start_point[-1]))
            idy2 = np.argmin(np.abs(np.array(flon) - end_point[0]))
            idx2 = np.argmin(np.abs(np.array(flat) - end_point[-1]))
            idx_s, idx_e = np.min([idx1, idx2]), np.max([idx1, idx2])
            idy_s, idy_e = np.min([idy1, idy2]), np.max([idy1, idy2])
            ref = ref[:, idx_s:idx_e, idy_s:idy_e]

        return ref, flat, flon, list(layer[:, 0, 0]), lats, lons

    def draw(self, ele_type, start_point=None, end_point=None):
        """
        ----------
        param: ele_type: 绘制的产品数据类型
        ----------
        """
        if ele_type == 'wind':
            dda = DDAWindController(self.fileinfo, self.level, self.jobid)
            logger.info("开始运行风场程序...")
            data_file = dda.run_exe()
            logger.info("开始读取数据绘图...")
            data, flat, flon, layer, xticks, yticks = self.get_wind_data(data_file[0])
            fig = go.Figure(data=[go.Surface(z=np.array(data[0]), x=flat[0][::-1, ::1], y=flon[0][::-1, ::1],
                                             colorscale=co_scale,
                                             cmin=0, cmax=70,
                                             colorbar=dict(tickmode='array',
                                                           tickvals=np.arange(0, 75, 5),
                                                           ticktext=levels,
                                                           tickfont=dict(size=18),
                                                           thickness=8,
                                                           len=0.8,
                                                           ticks="outside"))])
            layer = layer[0:layer.index(self.level) + 1]
            zh = 10  # 高度调整的基数
            for i in layer[1:]:
                idx = layer.index(i)
                h = i - layer[0]
                opacity = h / self.level  # 透明度值
                opacity = opacity if opacity > 0.3 else 0.3
                fig.add_traces(
                    data=[
                        go.Surface(z=np.array(data[idx]) + h * zh, x=flat[idx][::-1, ::1], y=flon[idx][::-1, ::1],
                                   colorscale=co_scale,
                                   cmin=0 + h * zh, cmax=70 + h * zh,
                                   showscale=False,
                                   opacity=1 - opacity)])
            self.update_layout(fig, xticks, yticks, layer, zh)
        else:
            ref = Reflectivity(self.fileinfo, self.jobid, self.ele, '三维图', ele_type)
            logger.info("开始读取数据...")
            data, flat, flon, layer, xticks, yticks = self.get_data(ref.data_file, start_point=start_point, end_point=end_point)

            fig = go.Figure(data=[go.Surface(z=np.array(data[2].T)[::2, ::2], x=(flat[::-1])[::2], y=flon[::2],
                                             colorscale=co_scale,
                                             cmin=0, cmax=70,
                                             colorbar=dict(tickmode='array',
                                                           tickvals=np.arange(0, 75, 5),
                                                           ticktext=levels,
                                                           tickfont=dict(size=18),
                                                           thickness=8,
                                                           len=0.8,
                                                           ticks="outside"))])
            def_value = list(np.arange(2500, 8000, 500))
            def_value.extend(list(np.arange(8000, 16000, 1000)))
            layer_def = def_value[0:def_value.index(self.level) + 1]
            zh = 20  # 高度调整的基数
            for i in layer_def[1:]:
                idx = layer.index(i)
                h = i - layer_def[0]
                opacity = (h / (layer_def[-1] - 1500))  # 透明度值
                opacity = opacity if opacity > 0.3 else 0.3
                fig.add_traces(
                    data=[
                        go.Surface(z=np.array(data[idx].T)[::2, ::2] + h * zh, x=(flat[::-1])[::2], y=flon[::2],
                                   colorscale=co_scale,
                                   cmin=0 + h * zh, cmax=70 + h * zh,
                                   showscale=False,
                                   opacity=1 - opacity)])
            self.update_layout(fig, xticks, yticks, layer_def, zh)
        return fig

    def update_layout(self, fig, lats, lons, layer, zh):
        # lats = np.arange(xtick[0], xtick[-1] + 0.01, (xtick[-1] - xtick[0]) / 4)
        # lons = np.arange(ytick[0], ytick[-1] + 0.01, (ytick[-1] - ytick[0]) / 4)
        fig.update_layout(autosize=False,
                          width=500,  # 长宽
                          height=600,
                          scene=dict(
                              yaxis=dict(tickmode='array',  # 选择了array 那么就是说要用tickvals 和 ticktext组合了。
                                         # range=[lons[0] - 0.2, lons[-1] + 0.2],
                                         tickvals=lons,  # 原标签值
                                         ticktext=[f'{round(lons[i], 2)}°E' for i in range(5)],  # 最终显示的值  (替换的值)
                                         tickfont=dict(size=12),
                                         showgrid=False,
                                         visible=True,
                                         # tickangle=-15,
                                         # autorange="reversed",
                                         title=' ',
                                         ),
                              xaxis=dict(tickmode='array',  # 选择了array 那么就是说要用tickvals 和 ticktext组合了。
                                         # range=[lats[0] - 0.2, lats[-1] + 0.2],
                                         tickvals=lats[::-1],  # 原标签值
                                         ticktext=[f'{round(lats[i], 2)}°N' for i in range(5)],  # 最终显示的值  (替换的值)
                                         tickfont=dict(size=12),
                                         showgrid=False,
                                         visible=True,
                                         # tickangle=45,
                                         title=' ',
                                         ),
                              zaxis=dict(tickmode='array',  # 选择了array 那么就是说要用tickvals 和 ticktext组合了。
                                         range=[0, (layer[-1] - layer[0] + 200) * zh] if zh == 20 else [0, layer[-1] * zh],
                                         tickvals=(np.array(layer) - layer[0]) * zh,  # 原标签值
                                         ticktext=[f'{i / 1000}km' for i in np.array(layer)],  # 最终显示的值  (替换的值)
                                         tickfont=dict(size=12),
                                         showgrid=True,
                                         visible=True,
                                         tickangle=0,
                                         title=' ',
                                         # titlefont=dict(size=25)
                                         ),
                              aspectratio=dict(x=0.7, y=0.7, z=1.8),  # 坐标x,y,z比例为0.8：0.8：1.5
                          ),
                          scene_camera_eye=dict(x=2, y=2, z=0.5),  # 调整视角 2, 1.5, 0.7
                          margin=dict(l=10, r=10, b=10, t=40),
                          title=dict(text=f"{np.around(layer[0] / 1000, 1)}km-{np.around(layer[-1] / 1000, 1)}km 垂直分布",  # 标题名称
                                     y=0.9,  # 位置，坐标轴的长度看做1
                                     x=0.4,
                                     xanchor='center',  # 相对位置
                                     yanchor='top',
                                     font=dict(size=15))
                          )
        # fig.update_coloraxes()
        fig.update_yaxes(automargin=True)  # Y-axis Title自动移到左边合适的位置

    def write_txt(self, dat, dim1, dim2, dim3, def_value=None, filters=2):
        """

        :param dat: 数据
        :param dim1: 高度层
        :param dim2: 纬度
        :param dim3: 经度
        :param def_value: 选择的高度层
        :param filters: 抽吸倍数
        :return: 文件路径。
        """
        if def_value is None:
            def_value = dim1
            # def_value = list(np.arange(2500, 8000, 500))
            # def_value.extend(list(np.arange(8000, 16000, 1000)))
        ele = self.ele.split('_')[0] if self.ele not in ['w', 'uv', '3D_structure'] else 'ref'
        idx_l = [[np.where(dim1 == i)[0][0] for i in def_value]]
        data = np.array(dat[idx_l[0]])[::1, ::2, ::2]
        lats = np.array(dim2)[::filters]
        lons = np.array(dim3)[::filters]
        data[data == -999] = np.nan
        if ele == 'scw':
            data[(data < 4) + (data > 5)] = 0
            data[data == 5] = 1
            data[data == 4] = 1
        fimename = os.path.join(self.datafile_path, 'data.txt')
        with open(fimename, 'w') as f:
            data_str = '{"data": ['
            for data_z in data:
                data_str += '['
                for data_x in data_z:
                    data_str += '['
                    for data_y in data_x:
                        data_str += "".join(f"{data_y:.6f},")
                    data_str = data_str[0:-1]
                    data_str += '],'
                data_str = data_str[0:-1]
                data_str += '],'
            data_str = data_str[0:-1]
            data_str += '],'
            lon_str = '"lon": ['
            lat_str = '"lat": ['
            for lon in lons:
                lon_str += "".join(f"{lon:.6f},")
            for lat in lats:
                lat_str += "".join(f"{lat:.6f},")
            lon_str = lon_str[0:-1]
            lon_str += '],'
            lat_str = lat_str[0:-1]
            lat_str += '],'
            height_str = f'"height": {str(def_value)},'
            if ele == 'scw':
                le = [0, 1]
                co = ['#a0a0a0', '#ff0000']
            elif ele == 'ref':
                le = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
                co = colormap[ele]["colors"]
            else:
                le = colormap[ele]["levels"]
                co = colormap[ele]["colors"]
            levels = f'"levels": {le},'
            colors = f'"colors": {co},'
            c_l_unicode = [ff.encode("unicode_escape") for ff in colormap[ele]["colorslabel"]]
            c_l = str(c_l_unicode).replace("b'", '"').replace("'", '"')
            colorslabel = f'"colorslabel": {c_l}'
            f.write(data_str.replace("nan", "null") + '\n' + lon_str + '\n' + lat_str + '\n' + height_str +
                    '\n' + levels + '\n' + colors + '\n' + colorslabel + '}')

        return fimename

    def run(self, ele_type=None, start_point=None, end_point=None, pic_format='html'):
        # if ele_type == 'wind':
        #     bottom = 1
        # elif ele_type == 'ref':
        #     bottom = 2.5
        # else:
        #     raise ValueError(f'不支持该{ele_type}雷达要素')
        # logger.info("开始处理...")
        # fig = self.draw(ele_type, start_point=start_point, end_point=end_point)
        # png_path = os.path.join(self.pic_path,
        #                         f'{ele_type}_3D_{bottom}_{np.around(self.level / 1000, 1)}.{pic_format}')
        # if pic_format == 'html':
        #     fig.write_html(png_path, include_plotlyjs=True)
        # elif pic_format == 'png':
        #     fig.write_image(png_path, engine="kaleido")  # "kaleido" "orca"

        if ele_type in ['vil', 'etp', 'crf']:
            raise ValueError(f'不支持该{ele_type}雷达要素')
        if ele_type == 'wind':
            dda = DDAWindController(self.fileinfo, self.level, self.jobid)
            logger.info("开始运行风场程序...")
            data_file = dda.run_exe()
            logger.info("开始读取数据...")
            data, flat, flon, layer, xticks, yticks = self.get_wind_data(data_file[0])

            data_file = self.write_txt(data, layer, flat[0, :, 0], flon[0, 0, :], def_value=layer, filters=1)
        else:
            ref = Reflectivity(self.fileinfo, self.jobid, self.ele, '三维图', ele_type)
            logger.info("开始读取数据...")
            data, flat, flon, layer, xticks, yticks = self.get_data(ref.data_file, start_point=start_point, end_point=end_point)

            def_value = list(np.arange(2500, 8000, 500))
            def_value.extend(list(np.arange(8000, 16000, 1000)))
            data_file = self.write_txt(data, layer, flat, flon, def_value=def_value, filters=2)

        # 上传图片
        # minio_client.put_object(data_file)
        pic_info = {"filename": os.path.basename(data_file), "path": transfer_path(data_file, is_win_path=True),
                    "element": self.ele}
        return [{"picFiles": pic_info}]

# if __name__ == '__main__':
#     import time
#     time_start = time.time()
#     file = [r'D:\tt\20220727\283359\Z_RADR_I_WEINN_20220727065100_O_DOR_MOC_CAP_FMT.nc']
#     S_3 = Spatial_3D(file, 1234, 8000)
#     S_3.run()
#     print(time.time() - time_start)
