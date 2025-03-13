import os
import re

import h5py
import numpy as np
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as sr
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap, Normalize
from mpl_toolkits.basemap import Basemap

from loguru import logger
from config.config import config_info

from service.colormap import colormap
from service.utils import prj_nom, transfer_path, wgs84toWebMercator, file_exit
# from utils.file_uploader import minio_client

base_path = os.path.dirname(os.path.dirname(__file__))  # os.getcwd()
obi_full_path = os.path.join(base_path, r'standard\FY4A_OBI_4000M_NOM_LATLON.HDF')
print(os.path.dirname(os.path.dirname(__file__)))
border_file = os.path.join(base_path, r'config\map\border.shp')  # 国界shp
province_file = os.path.join(base_path, r'config\map\province.shp')  # 省界shp
ten_line = os.path.join(base_path, r'config\map\ten_line.shp')  # 十段线shp
lat_0, lat_1, lon_0, lon_1 = 5, 54, 60, 139


class FY4A_analysis:

    def __init__(self, file, jobid, classes):
        """

        :param file: 文件信息
        :param jobid:
        :param classes: l1,l2 产品文件字段
        :param ele: 生成要素
        """
        # self.file = transfer_path(file[0][0]['filePath'])
        self.file = file_exit(file[0][0]['filePath'])
        self.jobid = jobid
        self.classes = classes
        self.ele = os.path.abspath(self.file).split('_')[-7][0:3]
        pic_path = config_info.get_FY4A_cfg['pic_path']
        self.pic_file_path = os.path.join(pic_path, classes, str(self.jobid))  # 生成图片保存的路径
        if not Path(self.pic_file_path).exists():
            Path.mkdir(Path(self.pic_file_path), parents=True, exist_ok=True)

    def set_cmap(self, colors, levels):
        """
        生成间断色卡
        :param colors: 颜色列表
        :param levels: 颜色对应数值
        :return: 标准cmap，norm
        """
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries=levels, ncolors=len(levels))
        return cmap, norm

    def get_smooth_cmap(self, cmap: str, normlim, colors=None):
        """
        生成自定义渐变色卡
        :param cmap: 内置色卡名称 当传入colors时无用
        :param normlim: 映射值范围[ , ]
        :param colors: 自定义色卡值，数组
        :return: 标准cmap，norm
        """
        if colors:
            cmap = LinearSegmentedColormap.from_list('my_color', colors, N=256)
        norm = Normalize(normlim[0], normlim[-1])
        return cmap, norm

    def draw_l1(self, border=False):
        ts = re.search(r"\d{14}_\d{14}", self.file).group().split('_')[0]
        time_str = (datetime.strptime(ts, '%Y%m%d%H%M%S') + timedelta(hours=8)).strftime('%Y%m%d%H%M%S')
        pic_info = []
        for k in ['NOMChannel01', 'NOMChannel09', 'NOMChannel12']:
            res = prj_nom(obi_full_path, self.file, k)
            f = h5py.File(self.file, 'r')
            cal_data = f[k.replace('NOM', 'CAL')][()]
            cal_data[cal_data > 60000] = 0
            res[res == 65535] = 0  # 填充值归0
            res = cal_data[res]

            fig = plt.figure()
            projection = ccrs.Mercator()
            ax = fig.add_subplot(projection=projection)
            m = Basemap(llcrnrlon=lon_0, llcrnrlat=lat_0, urcrnrlon=lon_1, urcrnrlat=lat_1, projection='merc', ax=ax)
            lon = np.arange(60, 139.01, 0.04)
            lat = np.arange(5, 54.01, 0.04)
            lon, lat = wgs84toWebMercator(lon, lat)
            Lon, Lat = np.meshgrid(lon, lat)
            # x, y = m(Lon, Lat)
            cmap, norm = self.get_smooth_cmap(colormap[k]['cmaps'], colormap[k]['levels'], colormap[k]['colors'])
            p = m.pcolormesh(Lon, Lat, res[::-1, :], cmap=cmap, norm=norm)
            plt.xlim(lon[0], lon[-1])
            plt.ylim(lat[0], lat[-1])
            if border:
                # bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                #                                     scale='10m', facecolor='none')
                # ax.add_feature(bodr, edgecolor='r', alpha=1, linewidth=1.)  # 添加国界线
                ax.coastlines(resolution='10m', alpha=1, linewidth=1.)  # 添加海岸线
                p_shapes = sr.Reader(province_file).geometries()
                ax.add_geometries(p_shapes, crs=ccrs.PlateCarree(), edgecolor='y', facecolor='none', lw=0.2)  # 添加省界
                b_shapes = sr.Reader(border_file).geometries()
                ax.add_geometries(b_shapes, crs=ccrs.PlateCarree(), edgecolor='r', facecolor='none', alpha=1, lw=0.4)  # 添加国界
                t_shapes = sr.Reader(ten_line).geometries()
                ax.add_geometries(t_shapes, crs=ccrs.PlateCarree(), edgecolor='r', facecolor='none', lw=0.4)  # 添加十段线

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.axis('off')
            savepath = os.path.join(self.pic_file_path, f'FY4A_L1_{k}_{time_str}.png')
            plt.savefig(savepath, transparent=True, bbox_inches="tight", pad_inches=0)
            logger.info(f"通道 {k} 绘制完成")
            # # minio_client.put_object(savepath)
            otherdata = {"gisPicLonLat": [(lon_0, lat_0), (lon_1, lat_1)]}
            pic_info.append({"filename": os.path.basename(savepath), "path": transfer_path(savepath, is_win_path=True),
                             "element": k, "otherdata": otherdata})

        return pic_info

    def draw_l2(self, cbar=False, border=False):
        ts = re.search(r"\d{14}_\d{14}", self.file).group().split('_')[0]
        time_str = (datetime.strptime(ts, '%Y%m%d%H%M%S') + timedelta(hours=8)).strftime('%Y%m%d%H%M%S')
        res = prj_nom(obi_full_path, self.file, self.ele)
        res[res == -999] = None
        if self.ele == 'CTT':
            res = res - 273.15
        elif self.ele == 'CTH':
            res = res / 1000

        resolution = 0.04
        fig = plt.figure()  # 图像尺寸figsize=((lon_1 - lon_0) / resolution / 100, (lat_1 - lat_0) / resolution / 100)
        # ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        projection = ccrs.Mercator()
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        m3 = Basemap(llcrnrlon=60, llcrnrlat=5, urcrnrlon=139, urcrnrlat=54, projection='merc', ax=ax)
        lon = np.arange(60, 139.01, 0.04)
        lat = np.arange(5, 54.01, 0.04)
        lon, lat = wgs84toWebMercator(lon, lat)
        Lon, Lat = np.meshgrid(lon, lat)
        # x, y = m3(Lon, Lat)  # cyl投影
        cmap, norm = self.set_cmap(colormap[self.ele]['colors'], colormap[self.ele]['levels'])
        im1 = m3.pcolormesh(Lon, Lat, res[::-1, :], cmap=cmap, norm=norm)
        plt.xlim(lon[0], lon[-1])
        plt.ylim(lat[0], lat[-1])
        if border:
            bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land',
                                                scale='10m', facecolor='none')
            ax.add_feature(bodr, edgecolor='r', alpha=1, linewidth=1.)  # 添加国界线
            ax.coastlines(resolution='10m', alpha=1, linewidth=1.)  # 添加海岸线
        if cbar:
            clb = plt.colorbar(im1, ax=ax, orientation="vertical", pad=0.03, shrink=0.6,
                               ticks=colormap[self.ele]['levels'], extend='both')
            clb.ax.set_title(colormap[self.ele]['unit'])
        plt.axis('off')
        savepath = os.path.join(self.pic_file_path, f'FY4A_L2_{self.ele}_{time_str}.png')
        logger.info(f"{self.ele}要素绘制完成")
        plt.savefig(savepath, transparent=True, bbox_inches="tight", pad_inches=0)
        # # minio_client.put_object(savepath)
        otherdata = {"gisPicLonLat": [(lon_0, lat_0), (lon_1, lat_1)]}
        pic_info = {"filename": os.path.basename(savepath), "path": transfer_path(savepath, is_win_path=True),
                    "element": self.ele, "otherdata": otherdata}

        return pic_info

    def run(self):
        pic_list = []
        if self.classes == 'l1':
            pic_list = self.draw_l1()
        elif self.classes == 'l2':
            pic_list = self.draw_l2()
        return [{"picFiles": pic_list}]


# if __name__ == '__main__':
#     filename_cth_l2 = r'E:\data\FY4A\tempFile\Z_SATE_C_BAWX_20230228002533_P_FY4A-_AGRI--_N_DISK_1047E_L2-_CTH-_MULT_NOM_20230228000000_20230228001459_4000M_V0001.NC'
#     filename_ctt_l2 = r'E:\data\FY4A\tempFile\Z_SATE_C_BAWX_20230228002540_P_FY4A-_AGRI--_N_DISK_1047E_L2-_CTT-_MULT_NOM_20230228000000_20230228001459_4000M_V0001.NC'
#     FY4A_analysis(filename_ctt_l2, 1234, 'l2', 'ctt').draw_l2(True, True)
