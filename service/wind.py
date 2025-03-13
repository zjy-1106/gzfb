import os
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from config.config import config_info
from service.utils import transfer_path, ax_add_time_range
# from utils.file_uploader import minio_client

mpl.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
mpl.use('Agg')
FONTSIZE = 12


class Wind_profile:
    def __init__(self, file_list, jobid, pic_type):
        self.jobid = jobid
        self.pic_type = pic_type
        self.wd = []
        self.ws = []
        self.w = []
        self.height = []
        self.time = []
        self.time_list = []
        pic_path = config_info.get_wind_cfg['pic_path']
        self.pic_file_path = os.path.join(pic_path, str(self.jobid))  # 生成图片保存的路径
        if not Path(self.pic_file_path).exists():
            Path.mkdir(Path(self.pic_file_path), parents=True, exist_ok=True)
        if file_list.__len__() == 1:
            raise ValueError("文件数量为1，不足以绘图")

        for file in file_list:
            file_path = transfer_path(file['filePath'])
            with open(file_path, mode='rb') as f:
                content_list = [i.strip() for i in f]
                x = content_list[1].split(b' ')
                SURF_CHN_BASIC_INFO_ID = x[0].decode()  # 区站号
                D_DATETIME = self.UTC_BJT(x[-1].decode())  # 观测时间     20190717151700
                # D_DATETIME = x[-1].decode()
                self.time.append(D_DATETIME)
                for txt in content_list[3:-1]:
                    # times.append(D_DATETIME)
                    self.time_list.append(datetime.datetime.strptime(D_DATETIME, "%Y%m%d%H%M%S"))
                    # 将文件中的‘/’, 转换成对应的‘9’
                    info = txt.decode().replace("/", '9').split()
                    self.height.append(info[0])  # 采样高度
                    self.wd.append(info[1])  # 水平风向
                    self.ws.append(info[2])  # 水平风速
                    self.w.append(info[3])  # 垂直风速
        rad = np.pi / 180.0
        self.wd = np.array(self.wd, dtype=float)
        self.ws = np.array(self.ws, dtype=float)
        self.w = np.array(self.w, dtype=float)
        self.height = np.array(self.height, dtype=float) / 1000
        self.time_list = np.array(self.time_list)
        self.u = - self.ws * np.sin(self.wd * rad)
        self.v = - self.ws * np.cos(self.wd * rad)
        self.u[self.u >= 999] = None
        self.v[self.v >= 999] = None
        self.w[self.w >= 999] = None
        self.ws[self.ws >= 999] = None

    def UTC_BJT(self, info):
        """
        :param info: 输入时间
        :return: 世界时转化为北京时
        """
        dt = datetime.datetime.strptime(info, "%Y%m%d%H%M%S")
        strdt = dt + datetime.timedelta(hours=8)
        return datetime.datetime.strftime(strdt, "%Y%m%d%H%M%S")

    def draw_uv(self, title, ylims=None, eff_time=None, hail_time=None):
        pic_list = []
        ws = self.ws.reshape(len(self.time), -1)
        if ws.shape[1] <= 1:
            raise ValueError(f'{self.time[0]}-{self.time[-1]}时间段内数据量不足, 需扩大时间范围')
        u = self.u.reshape(len(self.time), -1)
        v = self.v.reshape(len(self.time), -1)
        height = self.height.reshape(len(self.time), -1)
        time_list = self.time_list.reshape(len(self.time), -1)
        xtime = time_list[:, 0]
        yheight = height[0, :]

        fig, axs = plt.subplots(figsize=(8, 3))
        plt.rcParams['font.size'] = FONTSIZE

        cs = axs.contourf(xtime, yheight, ws.T, cmap="jet")
        cb = fig.colorbar(cs, ax=axs, fraction=0.027, pad=0.02, aspect=30)
        cb.ax.set_title('m/s', fontsize=8)
        cb.ax.tick_params(labelsize=8)
        # # cbar.set_ticks(np.linspace(0, 12, 7))
        barb_len = 5
        barb_line_width = 0.5
        xrange = round(len(self.time) / 30) if len(self.time) >= 40 else 1
        yrange = 2
        q = axs.barbs(time_list[::xrange, ::yrange], height[::xrange, ::yrange],
                      u[::xrange, ::yrange], v[::xrange, ::yrange],
                      length=barb_len, linewidth=barb_line_width,
                      sizes={"spacing": 0.2, 'height': 0.5, 'width': 0.5, 'emptybarb': 0}, zorder=1,
                      barb_increments=dict(half=2, full=4, flag=20))
        title_time = '~'.join([xtime[0].strftime('%Y-%m-%d %H:%M'), xtime[-1].strftime('%Y-%m-%d %H:%M')])
        axs.set_title(f'{title}\t{title_time}', fontsize=FONTSIZE)
        # # majorloc = mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21])
        majorformatter = DateFormatter('%H:%M')

        # ax.xaxis.set_major_locator(majorloc)
        axs.xaxis.set_major_formatter(majorformatter)
        axs.xaxis.set_minor_locator(mdates.MinuteLocator(interval=6))
        # axs.set_xlabel(xtime[0].strftime('%Y-%m-%d'))
        axs.set_ylabel('高度 (km)', fontsize=FONTSIZE)
        axs.set_ylim(ylims)
        axs.tick_params(axis="x", direction="out", which='major', length=8, width=1.5)
        axs.tick_params(axis="x", direction="out", which='minor', length=4, width=1.0)
        ax_add_time_range(axs, eff_time, alpha=0.6, color='w')
        ax_add_time_range(axs, hail_time, alpha=0.4, color='k')

        fig.align_ylabels(axs)
        plt.tick_params(labelsize=FONTSIZE)
        pic_file = os.path.join(self.pic_file_path,
                                f"{xtime[0].strftime('%Y%m%d%H%M%S')}-{xtime[-1].strftime('%Y%m%d%H%M%S')}_uv.png")

        fig.savefig(pic_file, dpi=200, bbox_inches='tight', pad_inches=0.1)
        # 上传图片
        # minio_client.put_object(pic_file)
        pic_info = {"filename": os.path.basename(pic_file),
                    "path": transfer_path(pic_file, is_win_path=True),
                    "element": "uv"}
        return pic_info

    def draw_w(self, title, ylims=None, eff_time=None, hail_time=None):
        pic_list = []
        w = self.w.reshape(len(self.time), -1)
        if w.shape[1] <= 1:
            raise ValueError(f'{self.time[0]}-{self.time[-1]}时间段内数据量不足, 需扩大时间范围')
        height = self.height.reshape(len(self.time), -1)
        time_list = self.time_list.reshape(len(self.time), -1)
        xtime = time_list[:, 0]
        yheight = height[0, :]
        value_max = max(abs(np.nanmin(w)), abs(np.nanmax(w)))

        fig, axs = plt.subplots(figsize=(8, 3))
        plt.rcParams['font.size'] = FONTSIZE
        cs2 = axs.contourf(xtime, yheight, w.T, cmap='RdYlBu_r', vmin=-value_max, vmax=value_max)
        title_time = '~'.join([xtime[0].strftime('%Y-%m-%d %H:%M'), xtime[-1].strftime('%Y-%m-%d %H:%M')])
        axs.set_title(f'{title}\t{title_time}', fontsize=FONTSIZE)
        cb = fig.colorbar(cs2, ax=axs, fraction=0.027, pad=0.02, aspect=30)
        cb.ax.set_title('m/s', fontsize=8)
        cb.ax.tick_params(labelsize=8)
        # cbar.set_ticks([-1.2, -0.6, 0, 0.6, 1.2])
        # cbar2.ax.tick_params(labelsize=14)
        # # majorloc = mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21])
        majorformatter = DateFormatter('%H:%M')

        # ax.xaxis.set_major_locator(majorloc)
        axs.xaxis.set_major_formatter(majorformatter)
        axs.xaxis.set_minor_locator(mdates.MinuteLocator(interval=6))
        # axs.set_xlabel(xtime[0].strftime('%Y-%m-%d'))
        axs.set_ylabel('高度 (km)', fontsize=FONTSIZE)
        axs.set_ylim(ylims)
        axs.tick_params(axis="x", direction="out", which='major', length=8, width=1.3)
        axs.tick_params(axis="x", direction="out", which='minor', length=4, width=1.1)  # , pad=8, labelsize=15,
        ax_add_time_range(axs, eff_time, alpha=0.6, color='w')
        ax_add_time_range(axs, hail_time, alpha=0.4, color='k')

        fig.align_ylabels(axs)
        plt.tick_params(labelsize=FONTSIZE)
        pic_file = os.path.join(self.pic_file_path,
                                f"{xtime[0].strftime('%Y%m%d%H%M%S')}-{xtime[-1].strftime('%Y%m%d%H%M%S')}_w.png")

        fig.savefig(pic_file, dpi=200, bbox_inches='tight', pad_inches=0.1)

        # minio_client.put_object(pic_file)
        pic_info = {"filename": os.path.basename(pic_file),
                    "path": transfer_path(pic_file, is_win_path=True),
                    "element": "w"}

        return pic_info

    def run(self, title, ylims=None, eff_time=None, hail_time=None):
        if self.pic_type == 'uv':
            pic_list = self.draw_uv(title, ylims, eff_time=eff_time, hail_time=hail_time)
        elif self.pic_type == 'w':
            pic_list = self.draw_w(title, ylims, eff_time=eff_time, hail_time=hail_time)
        else:
            raise Exception("输入正确绘图类型：uv，w")
        return [{"picFiles": pic_list}]


# if __name__ == '__main__':
#     # file_name = "E:/data/风廓线/20220504/WNDOBS/HOBS/Z_RADA_I_56691_20220504000000_P_WPRD_LC_HOBS.TXT"
#     path = 'E:/data/风廓线/20220504/WNDOBS/ROBS/'
#     name = pd.read_csv(path + '5669120220504WNDOBS.LOG', sep='\s+', header=None)  # 5669120220504WNDOBS.LOG
#     names = name.iloc[:, 1]
#     file_names = []
#     file_names.append([os.path.join(path, na) for na in names])
#     pic_type = 'uv'
#     pic_path = 'D:/测试/'
#     Wind_profile(file_names[0], pic_type, pic_path).run()
