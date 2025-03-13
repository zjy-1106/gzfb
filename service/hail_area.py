import os
import glob
import re
import itertools
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import scipy
from loguru import logger

import numpy as np
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.cm as cm
from geopy.distance import geodesic
from scipy.interpolate import splprep
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from config.config import config_info
from service.colormap import colormap
from service.dsd_rainrate import DsdRainRateController
from service.utils import transfer_path
# from utils.file_uploader import minio_client
from service.ref import Reflectivity

str_point = [103.01042080, 25.960774]  # 叠图区域
end_point = [104.836801443, 27.58676280]


def get_part(data, lons, lats, strat_point, end_point, filter_value=50):
    data = np.copy(data)
    data_lc = np.ma.masked_array(data, mask=~(
            (lons < max(strat_point[0], end_point[0])) & (lons > min(strat_point[0], end_point[0])) &
            (lats < max(strat_point[1], end_point[1])) & (lats > min(strat_point[1], end_point[1]))))
    fill_ids = np.argwhere(data_lc.mask == False)

    lon_index_str, lon_index_end = min(fill_ids[:, 1]), max(fill_ids[:, 1])
    lat_index_str, lat_index_end = min(fill_ids[:, 0]), max(fill_ids[:, 0])
    data_cen = np.nanmax(data_lc[lat_index_str:lat_index_end, lon_index_str:lon_index_end])
    filter_value = (np.around(data_cen / 5) - 2) * 5

    data_lc[lat_index_str:lat_index_end, lon_index_str] = filter_value
    data_lc[lat_index_str:lat_index_end, lon_index_end] = filter_value
    data_lc[lat_index_str, lon_index_str:lon_index_end] = filter_value
    data_lc[lat_index_end, lon_index_str:lon_index_end] = filter_value
    return data_lc, filter_value


def get_contour_verts(cn):
    contours = []
    idx = 0
    # for each contour line
    # print(cn.levels)
    for cc, vl in zip(cn.collections, cn.levels):
        # for each separate section of the contour line
        for pp in cc.get_paths():
            paths = {"id": idx, "type": 0, "value": float(vl)}
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append([float(vv[0][0]), float(vv[0][1])])  # vv[0] 是等值线上一个点的坐标，是 1 个 形如 array[12.0,13.5] 的 ndarray。
            paths["coords"] = xy
            contours.append(paths)
            idx += 1
    return contours


class Hail:
    def __init__(self, fileinfo, jobid):

        self.fileinfo = fileinfo
        self.jobid = jobid
        self.ele = 'crf'
        self.level = 0
        self.png_path = config_info.get_hail_cfg['pic_path']

    def draw_hor(self, draw_data, time, s_point=None, e_point=None):
        if e_point is None:
            e_point = []
        if s_point is None:
            s_point = []
        file_time = time.strftime('%Y%m%d%H%M')
        lons, lats = np.meshgrid(draw_data.get('lon'), draw_data.get('lat'))
        my_map = Basemap(projection='merc', llcrnrlat=str_point[1], urcrnrlat=end_point[1],
                         llcrnrlon=str_point[0], urcrnrlon=end_point[0])
        lons, lats = my_map(lons, lats)

        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = ListedColormap(colormap[self.ele]["colors"])
        norm = BoundaryNorm(boundaries=colormap[self.ele]["levels"], ncolors=len(colormap[self.ele]["levels"]))
        pic = my_map.pcolormesh(lons, lats, draw_data.get('data')[self.level, :, :],
                                norm=norm, cmap=cmap, shading='auto', alpha=0.0)
        data_l = np.copy(draw_data.get('data')[self.level, :, :])

        for k in range(len(s_point)):
            try:
                input_s = my_map(s_point[k][0], s_point[k][1])
                input_e = my_map(e_point[k][0], e_point[k][1])
                data_lc, filter_value = get_part(data_l, lons, lats, input_s, input_e)

                cs = my_map.contour(lons, lats, data_lc,
                                    levels=colormap[self.ele]["levels"], colors='k', linewidths=0.1,
                                    linestyles='-', zorder=0, alpha=0.0)
                test_vs = get_contour_verts(cs)

                line_xy = []
                res1 = [x for index, x in enumerate(test_vs) if
                        {k: v for k, v in x.items() if (k == "value" and v == filter_value)}]
                res1.sort(key=lambda s: len(s["coords"]), reverse=True)
                for ids, line_c in enumerate(res1):
                    if line_xy and len(line_c['coords']) > 4:
                        # 判断为闭合曲线后则不再添加其他点
                        if geodesic(my_map(line_xy[0][0], line_xy[0][1], inverse=True)[::-1],
                                    my_map(line_xy[-1][0], line_xy[-1][1], inverse=True)[::-1]).m < 400:
                            continue
                        r = []
                        for i_d in range(ids, len(res1)):
                            r.append(geodesic(my_map(line_xy[-1][0], line_xy[-1][1], inverse=True)[::-1],
                                              my_map(res1[i_d]['coords'][0][0], res1[i_d]['coords'][0][1], inverse=True)[::-1]).m)

                        ids_min = np.argmin(np.array(r))
                        temps = res1[ids]['coords']
                        res1[ids]['coords'] = res1[ids_min + ids]['coords']
                        res1[ids_min + ids]['coords'] = temps
                        line_xy.extend(res1[ids]['coords'])
                    elif len(line_c['coords']) > 4:
                        line_xy = line_c['coords']
                line_xy = np.array(line_xy)
                # tck, u = scipy.interpolate.splprep(line_xy.T, s=0, per=True)
                # unew = np.linspace(0, 1, 1000)
                # out = scipy.interpolate.splev(unew, tck)
                # # plt.cla()
                # my_map.scatter(out[0], out[1], s=1, color='w', marker='o')
                plt.fill(line_xy[:, 0], line_xy[:, 1], edgecolor='k', fill=False, lw=0.3)

            # x_lons = np.arange(lons[0, 0], lons[0, -1] + 0.0001, (lons[0, -1] - lons[0, 0]) / 4)
            # y_lats = np.arange(lats[0, 0], lats[-1, 0] + 0.0001, (lats[-1, 0] - lats[0, 0]) / 4)
            # plt.xticks(ticks=x_lons, labels=[f'{round(x_lons[i], 2)}°E' for i in range(5)])
            # fig.autofmt_xdate()
            # plt.yticks(ticks=y_lats, labels=[f'{round(y_lats[i], 2)}°N' for i in range(5)])
            # plt.title(f'区域内回波强度  高度：{draw_data.get("height")[self.level]}km')
            # br = plt.colorbar(pic, fraction=0.023, pad=0.02, ticks=colormap[self.ele]["levels"],
            #                   aspect=30)  # 依次调整colorbar大小，与子图距离，label，横纵比
            # br.ax.set_title('dBZ')
            except IndexError:
                logger.warning('没有单体信息')
        plt.axis('off')

        pic_file_path = os.path.join(self.png_path, time.strftime('%Y%m%d'), str(self.jobid))
        if not Path(pic_file_path).exists():
            Path.mkdir(Path(pic_file_path), parents=True, exist_ok=True)

        png_path = os.path.join(pic_file_path, f"fail_area_{file_time}.png")
        plt.savefig(png_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close()
        otherdata = {"args": None,
                     "gisPicLonLat": [str_point, end_point],
                     }
        # 上传图片
        # minio_client.put_object(png_path)
        pic_info = {"filename": os.path.basename(png_path), "path": transfer_path(png_path, is_win_path=True),
                    "element": "hail_area", "otherData": otherdata}
        return pic_info

    def run(self, point_dict=None):
        rf = Reflectivity([self.fileinfo], self.jobid, self.ele, '冰雹区域绘制', 6)
        pic_data, _ = rf.get_data()
        with Dataset(rf.data_file, 'r') as ds:
            file_time = datetime.fromtimestamp(ds.RadTime)
        df = pd.DataFrame.from_dict(point_dict)
        s_point = df.startPoint.tolist()
        e_point = df.endPoint.tolist()
        pic_info = self.draw_hor(pic_data, file_time, s_point, e_point)

        return [{"picFiles": [pic_info]}]
