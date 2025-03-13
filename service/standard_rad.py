# -*- coding: utf-8 -*-
import os
import re

from cinrad.constants import deg2rad
from loguru import logger
import cinrad
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.colors as cmx
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cinrad.visualize import PPI, Section
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.cm import ScalarMappable

from config.config import config_info
from service.utils import transfer_path, select_data, file_exit
from service.colormap import colormap
# from utils.file_uploader import minio_client

mpl.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
element_conv = {"ref": "REF", "wid": "SW", "vel": "VEL", "vil": "VIL", "crf": "CR", "etp": "ET", "kdp": "KDP", "zdr": "ZDR"}
element_reconv = {"REF": "ref", "SW": "wid", "VEL": "vel", "VIL": "vil", "CR": "crf", "ET": "etp", "KDP": "kdp", "ZDR": "zdr"}


class Standard_rad:

    def __init__(self, file, element, jobid, threshold=None, elevation=None, azimuth=None, radius=230):
        """

        param file: 文件绝对路径
        param elevation: 仰角度数
        param element: 绘制要素
        param radius: 绘制图像的范围大小，单位km
        """
        # self.file = file[0][0]["filePath"].replace('gzfb', 'X:')
        self.file = file_exit(file[0][0]["filePath"])  # 本地若不存在则查询是否上传到minio，文件上传可能存在延迟
        self.f = cinrad.io.StandardData(self.file)
        product = self.f.available_product(0)  # 获取产品信息
        if elevation:
            try:
                self.ele = np.where(np.array(self.f.el) == elevation)[0][
                    0]  # [0.5, 0.5, 1.5, 1.5, 2.4, 3.3, 4.3, 6.0, 9.9, 14.6, 19.5]
            except IndexError:
                raise Exception(f"没有{elevation}°仰角数据, 不属于{set(self.f.el)}其中之一")
        else:
            self.ele = 0
        # 色卡
        colors, levels, self.units = colormap[element]["colors"], colormap[element]["levels"], colormap[element][
            "label"]
        self.s_cmap, self.s_norm, self.s_cmap_smooth = self.get_colormap(colors, levels)
        self.azimuth = azimuth
        self.element = element_conv[element]
        self.threshold = threshold
        self.radius = radius
        # self.data_time = datetime.strptime(re.search(r'\d{14}', os.path.basename(self.file)).group(), "%Y%m%d%H%M%S")
        self.data_time = self.f.scantime + timedelta(hours=8)
        pic_path = config_info.get_std_rad_cfg['pic_path']
        self.pic_path = os.path.join(pic_path, str(jobid))  # 生成图片保存的路径
        if not Path(self.pic_path).exists():
            Path.mkdir(Path(self.pic_path), parents=True, exist_ok=True)

    def get_colormap(self, colors, levels):
        s_cmap = cmx.ListedColormap(colors)
        s_norm = cmx.BoundaryNorm(boundaries=levels, ncolors=len(levels))
        s_cmap_smooth = cmx.LinearSegmentedColormap.from_list('variable_map', colors, N=256)
        return s_cmap, s_norm, s_cmap_smooth

    def get_filename(self, plot_type):
        pic_file = "{}_{:.2f}_{}_{}.png".format(
            self.data_time.strftime("%Y%m%d%H%M%S"),
            self.azimuth if self.azimuth else self.f.el[self.ele],
            self.element,
            plot_type
        )
        return os.path.join(self.pic_path, pic_file)

    def ppi_plot(self, data, pic_file):
        fig = plt.figure(figsize=(10, 10), dpi=200)

        fig = PPI(data, fig=fig, dpi=200, add_city_names=False, coastline=False, plot_labels=False, style="transparent",
                  cmap=self.s_cmap, norm=self.s_norm, nlabel=self.s_norm.N)  # fig, style="white" / style="black"
        plt.axis('off')
        # 添加雷达圈状线
        # fig.plot_range_rings([50, 100, 150, 200, 230], color='black', linewidth=1.0)
        # for d in [50, 100, 150, 200, 230]:
        #     plt.text(d * 1000 - 10000, 1000, f'{d}km', fontdict={'size': 12, 'color': 'k'})
        # plt.axhline(y=41.97139, xmin=0.01, xmax=0.99, ls="-", c="k")  # 添加水平直线
        # plt.axvline(x=126.19666, ymin=0.01, ymax=0.99, ls="-", c="k")  # 添加垂直直线
        # 设置经纬度
        # liner = fig.geoax.gridlines(draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')  # 设置经纬度虚线
        # liner.top_labels = False
        # liner.right_labels = False
        # liner.xformatter = LONGITUDE_FORMATTER
        # liner.yformatter = LATITUDE_FORMATTER
        # liner.xlabel_style = {'size': 15, 'color': 'red', 'weight': 'bold'}  # 设置经纬度颜色字号等
        # liner.ylabel_style = {'size': 15, 'color': 'red', 'weight': 'bold'}
        fig(pic_file)
        # minio_client.put_object(pic_file)
        logger.info(f"生成图片：{pic_file}")
        return pic_file

    def rhi_plot(self, datas, azimuth, pic_file, title):
        plot_data = []
        for data in datas:
            azi = data.azimuth.data
            dis = data.distance.data
            values = data.variables.mapping[self.element].data
            re_value = cinrad.grid.resample(values, dis, azi, dis[0], 359)
            azimuth1 = azimuth * deg2rad
            idx = np.argmin(np.abs(re_value[2][:, 0] - azimuth1))
            plot_data.append(re_value[0][idx])
        re_dis = re_value[1][0]  # 更改后距离
        re_el = np.array(list(set(self.f.el)))  # 仰角信息
        x = re_dis.reshape(len(re_dis), 1) * np.cos(re_el.reshape(1, len(re_el)) * deg2rad)
        y = re_dis.reshape(len(re_dis), 1) * np.sin(re_el.reshape(1, len(re_el)) * deg2rad)
        plot_data = np.array(plot_data)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        ax.set(xlabel='距离(km)', ylabel='高度(km)')
        ele_data = plot_data.T
        im = ax.pcolormesh(x, y, ele_data, cmap=self.s_cmap, norm=self.s_norm, shading='auto')
        # plt.contourf(x, y, ele_data, 256, cmap=cmap, norm=norm)

        time_title = self.data_time.strftime("%Y-%m-%d %H:%M:%S")
        plt.title(f"{title} \n {time_title} {element_reconv[self.element]} RHI azm:{azimuth:.2f}° ")
        d1, d2 = np.where(~np.isnan(ele_data))
        if d1.size and d2.size:
            x_max = np.max(re_dis[d1] * np.cos(np.array(re_el[d2]) * deg2rad))
            y_max = np.max(re_dis[d1] * np.sin(np.array(re_el[d2]) * deg2rad))
            ax.set(xlim=(0, int(x_max) + 1), ylim=(0, int(y_max) + 1))
        levels = colormap[element_reconv[self.element]]["levels"]
        cb = fig.colorbar(im, ticks=levels, shrink=0.9, pad=0.035)
        cb.ax.set_title(self.units)

        plt.savefig(pic_file, bbox_inches="tight", pad_inches=0.1)
        # minio_client.put_object(pic_file)
        logger.info(f"生成图片：{pic_file}")
        return pic_file

    def vsc_plot(self, data, start, end, pic_file, title):
        # -----------RHI-两点剖面绘图----------------------------------------------------
        vcs = cinrad.calc.VCS(data)
        sec = vcs.get_section(start_cart=start, end_cart=end)  # 传入经纬度坐标
        # fig = Section(sec)

        rhi = sec[self.element]
        xcor = sec["x_cor"]
        ycor = sec["y_cor"]
        rmax = np.nanmax(rhi.values)
        plt.figure(figsize=(10, 5), dpi=300)
        # plt.grid(True, linewidth=0.50, linestyle="-.", color="white")
        cmap = self.s_cmap_smooth
        norm = cmx.Normalize(self.s_norm.vmin, self.s_norm.vmax)
        hlim = 15
        # pic = plt.pcolormesh(xcor, ycor, rhi, cmap=cmap, norm=norm, shading="auto")
        pic = plt.contourf(xcor, ycor, rhi, 256, cmap=cmap, norm=norm)
        plt.ylim(0, hlim)
        title += "RHI\n"
        title += "Start: {}°N {}°E ".format(sec.start_lat, sec.start_lon)
        title += "End: {}°N {}°E ".format(sec.end_lat, sec.end_lon)
        title += "Time: " + datetime.strptime(
            sec.scan_time, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y-%m-%d %H:%M ")
        title += "Max: {:.1f}".format(rmax)
        plt.title(title)
        lat_pos = np.linspace(sec.start_lat, sec.end_lat, 6)
        lon_pos = np.linspace(sec.start_lon, sec.end_lon, 6)
        tick_formatter = lambda x, y: "{:.2f}°N\n{:.2f}°E".format(x, y)
        ticks = list(map(tick_formatter, lat_pos, lon_pos))
        cor_max = xcor.values.max()
        plt.xticks(np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * cor_max, ticks)
        plt.ylabel("高度 (km)")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        br = plt.colorbar(sm, fraction=0.03, pad=0.035, aspect=30)  # , ticks=self.s_norm.boundaries.tolist()
        br.ax.set_title(self.units)
        plt.savefig(pic_file)
        # minio_client.put_object(pic_file)
        logger.info(f"生成图片：{pic_file}")
        return pic_file

    def run(self, plot_type, start=None, end=None, title=None):
        # ty = {1: 'TREF', 2: 'REF', 3: 'VEL', 4: 'SW_wid', 5: 'SQI, NAN', 6: 'CPA, NAN', 7: 'ZDR', 8: 'LDR, NAN',
        #       9: 'RHO, error', 10: 'PHI_pdp', 11: 'KDP_kdp',
        #       12: 'CP, NAN', 14: 'HCL_hcl', 15: 'CF, NAN', 16: 'SNRH, NAN', 17: 'SNRV, NAN', 32: 'Zc', 33: 'Vc',
        #       34: 'Wc', 35: 'ZDRc'}
        pic_list = []
        pic_file = self.get_filename(plot_type)
        if plot_type == "PPI":
            otherdata = {"args": [{"argsType": "elevation",
                                   "defaultValue": float(str(self.f.el[0])),
                                   "specialArgs": [float(str(self.f.el[dd])) for dd in self.f.angleindex_r],
                                   "unit": "°"}],
                         "polarPicLonLat": {"lon": float(self.f.stationlon),
                                            "lat": float(self.f.stationlat),
                                            "radius": self.radius * 1000}
                         }
        else:
            otherdata = None

        if plot_type == "PPI":
            if self.element == "ET":
                # ---回波顶高,组合反射率,垂直累计液态水----et,cr,vil---------------------
                data = cinrad.calc.quick_et([self.f.get_data(i, self.radius, 'REF') for i in self.f.angleindex_r])
            elif self.element == "VIL":
                data = cinrad.calc.quick_vil([self.f.get_data(i, self.radius, 'REF') for i in self.f.angleindex_r])
            elif self.element == "CR":
                data = cinrad.calc.quick_cr([self.f.get_data(i, self.radius, 'REF') for i in self.f.angleindex_r])
            else:
                try:
                    data = self.f.get_data(self.ele, self.radius, self.element)  # 选择反射率数据
                except ValueError:
                    data = self.f.get_data(self.ele + 1, self.radius, self.element)
            if self.threshold:
                ele = list(data.keys())[0]
                data[ele] = xr.DataArray(select_data(data[ele].values, ele, self.threshold), dims=["azimuth", "distance"])
            pic_file = self.ppi_plot(data, pic_file)
        elif plot_type == 'RHI':
            if self.element in ["SW", "VEL"]:
                data = [self.f.get_data(i, self.radius, self.element) for i in [1, 3, 4, 5, 6, 7, 8, 9, 10]]
            else:
                data = [self.f.get_data(i, self.radius, self.element) for i in self.f.angleindex_r]
            if not self.azimuth:
                raise Exception("请输入方位角")
            pic_file = self.rhi_plot(data, self.azimuth, pic_file, title)
        else:
            if self.element in ["SW", "VEL"]:
                data = [self.f.get_data(i, self.radius, self.element) for i in [1, 3, 4, 5, 6, 7, 8, 9, 10]]
                del data['RF']
            else:
                data = [self.f.get_data(i, self.radius, self.element) for i in self.f.angleindex_r]
            pic_file = self.vsc_plot(data, start, end, pic_file, title)
        pic_list.append({
            "element": element_reconv[self.element],
            "filename": os.path.basename(pic_file),
            "path": transfer_path(pic_file, is_win_path=True),
            "img_type": "figure",
            "otherData": otherdata
        })

        return [{"picFiles": pic_list}]
#     start_time = time.time()
#     var = "SW"
#     start_cart = (124, 41.5)
#     end_cart = (127, 41.9)
#     e = None  # 4.3
#     Standard_rad(fi, var, e).run('rhi', start_cart, end_cart)
#
#     print(time.time() - start_time)
