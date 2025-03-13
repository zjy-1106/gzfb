#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
解析微波辐射计数据，绘图
"""
import glob
import json
import os
import re
import shutil
import zipfile

from datetime import datetime, timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.dates import DateFormatter, rrulewrapper, RRuleLocator

from config.config import config_info
from mq.pubulisher import publish_temp_msg
from service.effect_analysis import set_x_time
from service.mwr_l import MwrLliners
from service.utils import transfer_path, temp_message, ax_add_time_range
# from utils.file_uploader import minio_client

mpl.use('Agg')
mpl.rc("font", family='DengXian')
mpl.rcParams['axes.unicode_minus'] = False
FONTSIZE = 12

png_info = {
    "cloud_bottom": ("minute", "m"),  # 云底高度
    "acc_water_vapour": ("minute", "g/m${^2}$"),  # 累计液态水含水量
    "lwc": ("minute", "g/m${^3}$"),  # 液态水含量
    "temperature": ("温度", "${^°}$C"),  # 温度
    "vapour": ("水汽密度", "g/m${^3}$"),  # 水汽密度
    "liquid": ("液态水含量", "g/m${^3}$"),  # 液态水含量
    "RH": ("相对湿度", "%"),  # 相对湿度

}

colormap = {
    # {"cmap": 'rainbow', "vmin": -45, "vmax": 45}
    'temperature': ('jet', -45, 45),
    'vapour': ('jet', 0, None),
    'liquid': ('jet', 0, None),
    'RH': ('jet', 0, 100)
}

line_info = {
    'cloud_bottom': "m",
    'acc_water_vapour': "mm",  # 累计液态水含量
    'lwc': "mm",  # 液态水含量
    "temperature": "°C",  # 温度
    "vapour": "g/m³",  # 水汽含量
    "liquid": "g/m³",  # 液态水含量
    "RH": "%",  # 相对湿度
}
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


class MWRController:
    def __init__(self, file_list, job_id, start_time, end_time, ele, height=None, data_time=None, eles=None):
        self.ori_file_list = sorted(file_list, key=lambda x: x.get("filePath"))
        file_info_df = pd.DataFrame.from_dict(self.ori_file_list)
        self.ori_filepath_list = file_info_df["filePath"].to_list()  # 源数据文件

        # 数据开始结束时间
        if start_time and end_time:
            self.start_time = datetime.strptime(start_time, '%Y%m%d%H%M%S')
            self.end_time = datetime.strptime(end_time, '%Y%m%d%H%M%S')
        else:
            file_date = datetime.strptime(os.path.basename(self.ori_file_list[0].get('filePath'))[:8], '%Y%m%d')
            self.start_time = file_date
            self.end_time = file_date + timedelta(days=1)
        self.data_time = data_time  # 廓线图数据时间
        self.jobid = job_id
        self.height = height  # 如果传入高度参数，则按照高度返回对应的二维数据给前端绘制折现图，否则只绘制二维数据的填色图
        self.element = ele if ele is not None else 'temperature'
        self.elements = eles
        path_cfg = config_info.get_mwr_config
        self.output_data_path = path_cfg["exe_out_path"]  # 输出路径
        self.pic_file_path = os.path.join(path_cfg["pic_path"], str(job_id))  # 生成图片保存的路径
        self.exe_path = path_cfg["exe_path"]  # os.path.join(BASE_PATH, 'exe', 'MWR', 'Microradimeter_min.exe') # exe路径
        self.temp_path = path_cfg["temp_path"]  # 解压中间路径
        self.data_file_list = []  # 算法生成数据文件
        self.pic_list = []  # 生成的图片
        # 文件数据记录
        self.file_time_list = []
        self.height_list = []
        self.level_number = []
        self.temp_list = []
        self.rh_list = []
        self.water_vapour = []
        self.Liqc = []  # 液态水含水量
        self.cloud_bottom = []
        self.acc_water_vapour = []
        self.LWC = []  # 液态水含量
        self.AI = []
        self.kj = []
        self.TTI = []
        self.Li = []
        self.SI = []
        self.LCL = []
        self.LFC = []
        self.EC = []
        self.BLH = []
        self.data_map = {
            'cloud_bottom': self.cloud_bottom, 'acc_water_vapour': self.acc_water_vapour, 'lwc': self.LWC,
            'AI': self.AI, 'kj': self.kj, 'TTI': self.TTI, 'Li': self.Li, 'SI': self.SI,
            'LCL': self.LCL, 'LFC': self.LFC, 'EC': self.EC, 'BLH': self.BLH,
            'temperature': self.temp_list, 'vapour': self.water_vapour, 'liquid': self.Liqc, 'RH': self.rh_list
        }

    def run_exe(self):
        for i, file_info in enumerate(self.ori_filepath_list):
            f = transfer_path(file_info)
            time_str = re.search(r"\d{10}", f).group()

            temp_dir = os.path.join(self.temp_path, str(self.jobid))
            if not os.path.isdir(temp_dir):
                os.makedirs(temp_dir)

            if os.path.getsize(f) == 0:
                logger.warning(f"{f}原始文件错误,数据为空")
                continue
            # 文件不存在，进行解压缩
            try:
                with zipfile.ZipFile(f) as fz:
                    fz.extractall(path=temp_dir)
            except zipfile.BadZipFile:
                raise FileExistsError(f"{os.path.basename(f)} 源文件错误")

            file_id = self.ori_file_list[i]['fileId']
            cp_m_file = glob.glob(os.path.join(temp_dir, "*CP_M.txt"))
            for run_f in cp_m_file:
                time_str = re.search(r"\d{14}", run_f).group()
                out_dir = os.path.join(self.output_data_path, time_str[0:8], time_str[8:10], time_str[10:])
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                out_file = glob.glob(os.path.join(transfer_path(out_dir, is_win_path=True, is_inline_path=True), '*.dat'))

                # 如果文件已执行，则直接读取数据文件
                if not out_file:
                    # 执行exe, 输入文件为三种微波辐射计数据，水汽含量、温度、湿度和云底高度的txt文件、各种天气动力和热力参数的txt文件和液态水含量的txt文件
                    cmd = f"{self.exe_path} {run_f} {out_dir}"
                    os.system(cmd)
                    # print(cmd)
                    out_file = glob.glob(os.path.join(out_dir, '*.dat'))
                    if not out_file:
                        logger.error("mwr执行{}失败".format(out_file))
                        continue
                    temp_file_info = temp_message(out_file, [self.ori_file_list[i]['fileId']])
                    publish_temp_msg("20", self.jobid, "MWR", temp_file_info)

                self.data_file_list.append(out_file[0])
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir)

    def get_data(self, title, analysis_type=None):
        # 返回折线图数据
        for out_file in self.data_file_list:
            with open(out_file, 'r') as f:
                while 1:
                    data = []
                    i = 0
                    for line in f:
                        data.append(line)
                        i = i + 1
                        if i == 8:
                            break
                    if len(data) < 8:
                        break

                    time_info = []
                    for inde_time, ti in enumerate(data[0].split()[3:]):
                        if len(ti) == 1:
                            time_info.append('0' + ti)
                        else:
                            time_info.append(ti)
                    time_info = ''.join(time_info)

                    curr_time = datetime.strptime(time_info, '%Y%m%d%H%M')  # 数据时间
                    if curr_time < self.start_time:
                        continue
                    if curr_time > self.end_time:
                        break
                    self.file_time_list.append(curr_time)
                    self.level_number = float(data[1].split()[-1])
                    height_list = data[2].split()[2:]  # 83个高度层
                    if not self.height_list:  # 高度层数据只保留一份
                        self.height_list = [float(height) for height in height_list]
                    temp_list = data[3].split()[1:]  # 温度
                    self.temp_list.append(temp_list)
                    rh_list = data[4].split()[2:]  # 湿度
                    self.rh_list.append(rh_list)
                    water_vapour = data[5].split()[2:]  # 水汽含量
                    self.water_vapour.append(water_vapour)
                    liqc = data[6].split()[2:]  # 液态水含水量
                    self.Liqc.append(liqc)
                    cloud_bottom_height, acc_water_vapour, total_lwc = data[7].split()[-3:]  # 云底高度，累计水汽含量，液态水含量
                    self.cloud_bottom.append(cloud_bottom_height)
                    self.acc_water_vapour.append(acc_water_vapour)
                    self.LWC.append(total_lwc)
                    # AI, kj, TTI, Li, SI, LCL, LFC, EC, BLH = data[7].split()[-9:]
                    # self.AI.append(AI)
                    # self.kj.append(kj)
                    # self.TTI.append(TTI)
                    # self.Li.append(Li)
                    # self.SI.append(SI)
                    # self.LCL.append(LCL)
                    # self.LFC.append(LFC)
                    # self.EC.append(EC)
                    # self.BLH.append(BLH)
        if not self.file_time_list:
            raise Exception("源数据无当前时间段数据")

        if analysis_type == 1:
            # 廓线图数据
            file_time = [i.strftime('%Y%m%d%H%M%S') for i in self.file_time_list]
            time_inx = file_time.index(self.data_time) if self.data_time else 0
            x = np.around(np.array(self.data_map.get(self.element)[time_inx], dtype=float), 2).tolist()

            ret_data = [{
                'x': x,
                'y': [self.height_list],
                'xlabel': line_info.get(self.element),
                'ylabel': 'km',
                'yname': title,
                'time': str(self.file_time_list[time_inx]),
                "dataTimeList": list(set(file_time))
            }]
        else:
            # 折线图数据
            y_data = np.array(self.data_map.get(self.element), dtype=float)
            y_data[y_data <= -999.000] = None
            y_data = np.around(y_data, 2)
            ret_data = [{
                'x': [f'{t:%Y-%m-%d %H:%M}' for t in self.file_time_list],
                'y': [y_data.tolist()],
                'xlabel': '时间',
                'ylabel': line_info.get(self.element, ''),
                'yname': [title]
            }]

        return ret_data

    def draw(self, ele, ele_data, title, ylims=None, iscorrect=False, eff_time=None, hail_time=None):
        # 绘制填色图
        file_time = np.array(self.file_time_list)
        if file_time.size <= 1:
            raise ValueError("时段数据缺失")
        height_list = np.array(self.height_list)
        x_label = [f"{t:%H:%M}" for t in file_time]
        data = np.array(ele_data, dtype='float32').T

        data[data == -999.0] = np.nan
        if iscorrect and ele in ['RH', 'temperature']:
            title = f'{title}\t{png_info.get(ele)[0]}'
            fi = [
                {'filePath': r'X:\csi-fs\TQFile\LBR\20230804\LBR_20230804080000.txt'},
                {'filePath': r'X:\csi-fs\TQFile\LBR\20230804\LBR_20230804200000.txt'},
                {'filePath': r'X:\csi-fs\TQFile\LBR\20230810\LBR_20230810080000.txt'}]

            filename = MwrLliners(fi).run()
            with open(filename, 'r') as f:
                rmse = json.loads(f.readlines()[0])
            for j, i in enumerate(range(2000, 12500, 500)):
                h = height_list * 1000 + 2496.89  # 转换为海拔高度 m
                a = np.argwhere(h > i)[:, 0]
                b = np.argwhere(h < i + 500)[:, 0]
                ids = list(set(a).intersection(set(b)))
                data[ids] = data[ids] - rmse[self.element][j]

        data[data == -999.0] = np.nan
        fig, ax = plt.subplots(figsize=(8, 3))
        plt.rcParams['font.size'] = FONTSIZE
        if ele in ['RH', 'temperature']:
            cmp, vmin, vmax = colormap.get(ele)
        else:
            cmp, vmin, _ = colormap.get(ele)
            data[data < 0] = 0
            # vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        cm = plt.cm.get_cmap(cmp)
        # X, Y = np.meshgrid(file_time, height_list)
        ax0 = ax.pcolormesh(file_time, height_list, data, cmap=cm, vmin=vmin, vmax=vmax, shading='auto')

        C = ax.contour(file_time, height_list, data, 6, colors='black', linewidths=0.5)  # 画等值线
        ax.clabel(C, inline=True, fontsize=8, fmt='%.0f')
        ax.set_ylabel('高度(Km)', fontsize=FONTSIZE)
        ax.set_ylim(ylims)
        # ax.set_xlabel(f'{file_time[0]:%Y-%m-%d}')
        _, label = png_info.get(ele)  # 图片标注信息

        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

        # 避免时间轴密集
        # ratio = 1
        # if len(x_label) > 10:
        #     ratio = round(len(x_label) / 10)
        # ax.set_xticks(file_time[::ratio])
        # ax.set_xticklabels(x_label[::ratio], rotation=45)
        kwargs_major, kwargs_minor, timestyle = set_x_time(file_time)
        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)

        fig.autofmt_xdate(rotation=45)
        br = plt.colorbar(ax0, ax=ax, fraction=0.027, pad=0.03, aspect=30)
        if ele == 'RH':
            br.ax.set_yticks(np.arange(vmin, vmax + 1, 10))
            br.ax.set_yticklabels(['{:.0f}%'.format(x) for x in np.arange(vmin, vmax + 1, 10)])
        else:
            br.ax.set_title(label, fontsize=8)
        br.ax.tick_params(labelsize=8)
        plt.tick_params(labelsize=FONTSIZE)

        title_time = '~'.join([f'{file_time[0]:%Y-%m-%d %H:%M}', f'{file_time[-1]:%Y-%m-%d %H:%M}'])
        ax.set_title(f"{title}\t{title_time}")
        plt.tight_layout()
        img_file = os.path.join(self.pic_file_path, f'{file_time[0]:%Y%m%d%H%M}_{ele}.png')
        plt.savefig(img_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        # 上传图片
        # minio_client.put_object(img_file)
        pic_info = {"element": ele, "fileName": os.path.basename(img_file),
                    "path": transfer_path(img_file, is_win_path=True)}
        return pic_info

    def draw_sum(self, fig, ax, ele, ele_data, ylims=None, iscorrect=False, eff_time=None, hail_time=None):
        # 绘制填色图
        # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(7, 5), dpi=300, sharex="all")
        file_time = np.array(self.file_time_list)
        height_list = np.array(self.height_list)
        x_label = [f"{t:%H:%M}" for t in file_time]
        data = np.array(ele_data, dtype='float32').T

        data[data == -999.0] = np.nan
        if iscorrect and ele in ['RH', 'temperature']:
            fi = [
                {'filePath': r'X:\csi-fs\TQFile\LBR\20230804\LBR_20230804080000.txt'},
                {'filePath': r'X:\csi-fs\TQFile\LBR\20230804\LBR_20230804200000.txt'},
                {'filePath': r'X:\csi-fs\TQFile\LBR\20230810\LBR_20230810080000.txt'}]

            filename = MwrLliners(fi).run()
            with open(filename, 'r') as f:
                rmse = json.loads(f.readlines()[0])
            for j, i in enumerate(range(2000, 12500, 500)):
                h = height_list * 1000 + 2496.89  # 转换为海拔高度 m
                a = np.argwhere(h > i)[:, 0]
                b = np.argwhere(h < i + 500)[:, 0]
                ids = list(set(a).intersection(set(b)))
                data[ids] = data[ids] - rmse[self.element][j]

        data[data == -999.0] = np.nan
        # fig, ax = plt.subplots(figsize=(8, 3))
        if ele in ['RH', 'temperature']:
            cmp, vmin, vmax = colormap.get(ele)
        else:
            cmp, vmin, _ = colormap.get(ele)
            data[data < 0] = 0
            # vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        cm = plt.cm.get_cmap(cmp)
        # X, Y = np.meshgrid(file_time, height_list)
        ax0 = ax.pcolormesh(file_time, height_list, data, cmap=cm, vmin=vmin, vmax=vmax, shading='auto')

        C = ax.contour(file_time, height_list, data, 6, colors='black', linewidths=0.5)  # 画等值线
        ax.clabel(C, inline=True, fontsize=8, fmt='%.0f')
        ax.set_ylabel('高度(Km)', fontsize=FONTSIZE)
        ax.set_ylim(ylims)
        # ax.set_xlabel(f'{file_time[0]:%Y-%m-%d}')
        label, title = png_info.get(ele)  # 图片标注信息
        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

        # 避免时间轴密集
        # ratio = 1
        # if len(x_label) > 10:
        #     ratio = round(len(x_label) / 10)
        # ax.set_xticks(file_time[::ratio])
        kwargs_major, kwargs_minor, timestyle = set_x_time(file_time)
        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)

        fig.autofmt_xdate(rotation=45)
        br = plt.colorbar(ax0, ax=ax, fraction=0.023, pad=0.03, aspect=30)
        if ele == 'RH':
            br.ax.set_yticks(np.arange(vmin, vmax + 1, 10))
            br.ax.set_yticklabels(['{:.0f}%'.format(x) for x in np.arange(vmin, vmax + 1, 10)])
        else:
            br.ax.set_title(title, fontsize=8)
        br.ax.tick_params(labelsize=8)
        br.set_label(label=label, fontsize=8, loc='center')
        plt.tick_params(labelsize=FONTSIZE)

    def get_height_data(self, ele_data_list):
        # 获取二维数据对应高度层数据
        try:
            idx = self.height_list.index(self.height / 1000)
        except ValueError:
            raise Exception(f"高度参数错误，无该高度层数据 {self.height}")
        pic_data = [np.around(np.array(ele_data_list, dtype=float)[:, idx], 2).tolist()]
        return pic_data

    def run(self, title, analysis_type=None, ylims=None, station_and_equip_name=None, product_names=None,
            iscorrect=False, eff_time=None, hail_time=None):
        # if not os.path.isdir(self.output_data_path):
        #     os.makedirs(self.output_data_path)
        if not os.path.isdir(self.pic_file_path):
            os.makedirs(self.pic_file_path)

        self.run_exe()
        # 绘制填色图
        pic_list = []
        pic_data = self.get_data(title)
        ele_name_list = ['temperature', 'vapour', 'liquid', 'RH']
        ele_list = (self.temp_list, self.water_vapour, self.Liqc, self.rh_list)
        if self.element in ele_name_list:
            # 只处理传入要素
            if analysis_type == 1:
                # 获取廓线图数据
                pic_data = self.get_data(title, analysis_type)
                return [{"picFiles": pic_list, "picData": pic_data}]
            elif analysis_type == 5:
                idx = ele_name_list.index(self.element)
                if self.height is None:
                    self.height = self.height_list[0] * 1000
                ele_data = self.get_height_data(ele_list[idx])
                pic_data = [{
                    'x': [f'{t:%Y-%m-%d %H:%M}' for t in self.file_time_list],
                    'y': ele_data, 'ylabel': line_info[self.element], 'yname': title
                }]
                return [{"picFiles": pic_list, "picData": pic_data}]

            if self.elements:
                # 绘制多个要素
                if len(self.elements) == 1:
                    pic_list.append(self.draw(self.element, self.data_map.get(self.element), title, ylims,
                                              iscorrect=iscorrect, eff_time=eff_time, hail_time=hail_time))
                else:
                    fig, axs = plt.subplots(len(self.elements), 1, figsize=(8, 9), dpi=300, sharex="all")
                    plt.rcParams['font.size'] = FONTSIZE
                    file_time = np.array(self.file_time_list)
                    for i, ax in enumerate(axs):
                        self.draw_sum(fig, ax, self.elements[i], self.data_map.get(self.elements[i]), ylims,
                                      iscorrect=iscorrect, eff_time=eff_time, hail_time=hail_time)
                        title_time = '~'.join([f'{file_time[0]:%Y-%m-%d %H:%M}', f'{file_time[-1]:%Y-%m-%d %H:%M}'])
                        if i < 1:
                            ax.set_title(f"{station_and_equip_name}\t{title_time}", fontsize=FONTSIZE)
                        # else:
                        #     ax.set_title(f"{product_names[i]}")
                    plt.tight_layout()
                    temp_name = '_'.join([nam for nam in self.elements])
                    img_file = os.path.join(self.pic_file_path, f'{temp_name}_{file_time[0]:%Y%m%d%H%M}.png')
                    plt.savefig(img_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    # 上传图片
                    # minio_client.put_object(img_file)
                    pic_info = {"element": temp_name, "fileName": os.path.basename(img_file),
                                "path": transfer_path(img_file, is_win_path=True)}
                    pic_list.append(pic_info)
            else:
                if iscorrect:
                    for ele in ['temperature', 'RH']:
                        pic_list.append(
                            self.draw(ele, self.data_map.get(ele), title, ylims, iscorrect=iscorrect,
                                      eff_time=eff_time, hail_time=hail_time))
                else:
                    pic_list.append(
                        self.draw(self.element, self.data_map.get(self.element), title, ylims, iscorrect=iscorrect,
                                  eff_time=eff_time, hail_time=hail_time))

            # 如果指定高度参数，则二维数据返回对应高度数据给前端
            if self.height is not None:
                idx = ele_name_list.index(self.element)
                ele_data = self.get_height_data(ele_list[idx])
                pic_data = [{
                    'x': [f'{t:%Y-%m-%d %H:%M}' for t in self.file_time_list],
                    'y': ele_data, 'ylabel': line_info[self.element], 'yname': title
                }]
        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, self.data_file_list)
        return [{"picFiles": pic_list, "picData": pic_data}]

# if __name__ == '__main__':
#     filelist = [{'filePath': "D:/Documents/gzfb/mwr/20220720/20220720_csv.zip", 'fileId': 19000}]
#     mwr = MWRController(filelist, 1, '', '', height=2)
#     print(mwr.run())
