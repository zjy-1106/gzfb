# -*- coding: utf-8 -*-
import glob
import os
import re
import time

from pathlib import Path

import matplotlib
import numpy as np
from datetime import datetime
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

from config.config import config_info, c_pic_config, c_line_config, c_profile_config
from module.public.cwr_to_nc import CWRData_nc
from mq.pubulisher import publish_temp_msg
from service.draw import draw_color_image
from service.utils import merge_data, transfer_path, temp_message, list_to_array

"""
C波段连续波雷达算法程序
"""

matplotlib.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 6
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


class CWRData:
    """读取C波段雷达算法程序生成的文件"""

    def __init__(self, data_files, start_time, end_time):
        self.start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
        self.end_time = datetime.strptime(end_time, "%Y%m%d%H%M%S")
        self.time_list = []  # 数据时间列表
        self.lib_num = []  # 数据库数
        self.lib_length = []  # 数据库长
        self.raw_ref = []  # 原始回波强度
        self.raw_vel = []  # 原始径向速度
        self.raw_sw = []  # 原始速度谱宽
        self.qc_ref = []  # 质控后回波强度
        self.qc_vel = []  # 质控后径向速度
        self.qc_sw = []  # 质控后速度谱宽
        self.vair = []  # 反演的空气上升速度
        self.cloud_layer = []  # 云层数
        self.cloud_bottom = []  # 云底
        self.cloud_top = []  # 云顶
        self.bright_belt_bottom_height = []  # 零度层亮带的底高
        self.bright_belt_middle_height = []  # 零度层亮带的中间高
        self.bright_belt_top_height = []  # 零度层亮带的顶高
        self.qc_sz_distance_library_number = []  # 距离库数（质控回波强度谱宽密度）
        self.qc_sz_radial_velocity = []  # 有效回波强度谱密度对应的径向速度(向下为主)
        self.qc_sz_wide_spectral_density = []  # 有效回波强度谱密度

        data_files_temp = list(set(data_files))
        data_files_temp.sort(key=data_files.index)
        for file in data_files_temp:
            qc_sz_distance_library_number_tmp = []
            qc_sz_radial_velocity_tmp = []
            qc_sz_wide_spectral_density_tmp = []
            if os.path.isfile(file):
                with open(file, "r") as f:
                    for line in f:
                        if "Data and Time:" in line:
                            curr_time = datetime.strptime(line.split(":")[1][:14], "%Y%m%d%H%M%S")
                            if curr_time < self.start_time:
                                continue
                            if curr_time > self.end_time:
                                break
                            self.time_list.append(datetime.strptime(line.split(":")[1][:14], "%Y%m%d%H%M%S"))
                            # self.time_list.append(datetime.strptime(os.path.basename(file)[:14], "%Y%m%d%H%M%S"))
                            continue
                        elif "Bin num and Gatewidth(m):" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.lib_num.append(int(float(data[-2])))
                            self.lib_length.append(int(float(data[-1])))
                            continue
                        elif "Raw ref" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.raw_ref.append(data[2:])
                            continue
                        elif "Raw vel" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.raw_vel.append(data[2:])
                            continue
                        elif "Raw  sw" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.raw_sw.append(data[2:])
                            continue
                        elif "QC  ref" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.qc_ref.append(data[2:])
                            continue
                        elif "QC  vel" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.qc_vel.append(data[2:])
                            continue
                        elif "QC   sw" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.qc_sw.append(data[2:])
                            continue
                        elif "Vair" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.vair.append(data[1:])
                            continue
                        elif "Cloud layer and cloud boundary" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.cloud_layer.append(data[5])
                            # 云层为0时，没有云底和云顶
                            if int(float(data[5])) == 0:
                                self.cloud_bottom.append([np.nan])
                                self.cloud_top.append([np.nan])
                            else:
                                cloud_bottom_t = []
                                cloud_top_t = []
                                for i in range(int(float(data[5]))):
                                    cloud_bottom_t.append(data[6 + i * 2])
                                    cloud_top_t.append(data[7 + i * 2])
                                self.cloud_bottom.append(cloud_bottom_t)
                                self.cloud_top.append(cloud_top_t)
                            continue
                        elif "Bright band height" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.bright_belt_bottom_height.append(data[4])
                            self.bright_belt_middle_height.append(data[5])
                            self.bright_belt_top_height.append(data[6])
                            continue
                        elif "QC SZ" in line:
                            if curr_time and curr_time < self.start_time:
                                continue
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            qc_sz_distance_library_number_tmp.append(data[2])
                            data_num = int(float(data[3]))  # 回波强度谱密度有效数据的个数
                            qc_sz_radial_velocity_tmp.append(data[4:4 + data_num])
                            qc_sz_wide_spectral_density_tmp.append(data[4 + data_num:])
                            continue

            self.qc_sz_distance_library_number.append(qc_sz_distance_library_number_tmp)
            self.qc_sz_radial_velocity.append(qc_sz_radial_velocity_tmp)
            self.qc_sz_wide_spectral_density.append(qc_sz_wide_spectral_density_tmp)


class CWRController:
    def __init__(self, input_filelist, height, jobid, element, title, analysis_type, ylims=None, is_pic=False,
                 data_time=None, start_time=None, end_time=None):
        self.time_list = None
        if start_time is None and data_time:
            self.start_time = data_time
            self.end_time = data_time
        else:
            self.start_time = start_time
            self.end_time = end_time
        self.title = title
        file_info = sorted(input_filelist, key=lambda x: x.get("filePath"))
        file_info_df = pd.DataFrame.from_dict(file_info)
        self.ori_file_list = file_info_df["filePath"].to_list()  # 源数据文件
        self.fileid = file_info_df["fileId"].to_list()
        self.ori_file_list.sort()
        self.height = height
        self.element = element
        self.jobid = jobid
        self.analysis_type = analysis_type
        self.ylims = ylims
        self.is_pic = is_pic if is_pic is not None else True
        self.data_time = data_time

        path_cfg = config_info.get_cwr_config
        self.output_data_path = path_cfg["exe_out_path"]  # 输出路径
        self.output_ncdata_path = path_cfg["nc_out_path"]  # nc文件存储路径
        self.pic_file_path = path_cfg["pic_path"]  # 生成图片保存的路径
        if '202402060000' < self.start_time < '202403010000':  # 2024-02-06时C波段数据更改 库数：500到800，谱点数：512到256
            self.exe_path = path_cfg["exe_path_new"]  # os.path.join(BASE_PATH, 'exe', 'CWR', 'CWR2024_2.exe')  exe路径
        else:
            self.exe_path = path_cfg["exe_path"]  # os.path.join(BASE_PATH, 'exe', 'CWR', 'CWR2.exe')
        self.data_path = path_cfg["origin_data"]  # 源数据路径
        self.minio_disk = path_cfg["minio_disk"]  # mini路径需要替换为本地盘符

        self.output_files = []  # 算法程序生成的文件
        self.output_files_sz = []

    @staticmethod
    def get_filetime(filename):
        return re.search(r"(?<=CWR)\d{14}", filename).group()

    def get_filedate(self, filenames):
        time_day_s = datetime.strptime(re.search(r"\d{14}", filenames[0]).group()[0:10], '%Y%m%d%H')
        time_day_d = datetime.strptime(re.search(r"\d{14}", filenames[-1]).group()[0:10], '%Y%m%d%H')
        time_hour = pd.date_range(time_day_s, time_day_d, freq='H')
        return time_hour.values

    def run_exe(self):
        """"构造input_file，运行算法exe程序"""
        if not os.path.isdir(self.output_data_path):
            os.makedirs(self.output_data_path)
        if not os.path.isdir(self.pic_file_path):
            os.makedirs(self.pic_file_path)
        # 算法程序输出数据文件保存路径, 后续改为时间段命名的文件夹
        self.ori_file_list.sort()

        time_day = re.search(r"\d{14}", self.ori_file_list[0]).group()[0:8]

        if not self.is_pic:
            if len(self.ori_file_list) > 1200:
                output_path = os.path.join(self.output_data_path, time_day)  # 一天的数据文件
                file_label = time_day
            else:
                time_day_hour = re.search(r"\d{14}", self.ori_file_list[0]).group()[8:10]
                output_path = os.path.join(self.output_data_path, time_day, time_day_hour)  # 一小时的数据文件，则存入固定位置。
                file_label = time_day + time_day_hour
        else:
            file_label = f'{self.get_filetime(self.ori_file_list[0])}-{self.get_filetime(self.ori_file_list[-1])}'
            output_path = os.path.join(self.output_data_path, file_label)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        input_file_dic = {}

        time_days = self.get_filedate(self.ori_file_list)
        for ti_v in time_days:
            # 判断一天的数据是否存在。
            nc_path = os.path.join(transfer_path(self.output_ncdata_path, is_win_path=True, is_inline_path=True),
                                   str(ti_v).split('T')[0].replace('-', ''))
            dat_path = os.path.join(transfer_path(self.output_data_path, is_win_path=True, is_inline_path=True),
                                    str(ti_v).split('T')[0].replace('-', ''))
            dat_hour_path = os.path.join(transfer_path(self.output_data_path, is_win_path=True, is_inline_path=True),
                                         str(ti_v).split('T')[0].replace('-', ''), str(ti_v).split('T')[-1][0:2])
            dat_paths = glob.glob(os.path.join(dat_path, '*1.dat'))
            dat_hour_paths = glob.glob(os.path.join(dat_hour_path, '*1.dat'))
            if dat_paths:
                self.output_files.append(dat_paths[0])
                # self.output_files_sz.append(glob.glob(os.path.join(dat_path, '*2.dat'))[0])
                logger.info(f"文件{self.output_files[-1]}存在，无需执行exe")
            elif dat_hour_paths:
                self.output_files.append(dat_hour_paths[0])
                # self.output_files_sz.append(glob.glob(os.path.join(dat_hour_path, '*2.dat'))[0])
                logger.info(f"文件{self.output_files[-1]}存在，无需执行exe")
            else:
                self.output_files = []
                self.output_files_sz = []
                break
        if self.output_files:
            return
        output_path_inline = transfer_path(output_path, is_win_path=True, is_inline_path=True)
        output_paths_inline = glob.glob(os.path.join(output_path_inline, '*1.dat'))
        if output_paths_inline:  # 远程
            # logger.info("开始拉取文件")
            self.output_files.append(output_paths_inline[0])
            # self.output_files_sz = file_exit(glob.glob(os.path.join(output_path_inline, '*2.dat')))
            # logger.info("拉取完成")
            return
        output_paths = glob.glob(os.path.join(output_path, '*1.dat'))
        if output_paths:  # 本地
            self.output_files.append(output_paths[0])
            # self.output_files_sz.append(glob.glob(os.path.join(output_path, '*2.dat'))[0])
            return

        # 构建inputfile
        logger.info("开始构建txt")
        for f in self.ori_file_list:
            input_file_path = os.path.dirname(f)
            input_file = transfer_path(f)
            # if not os.path.isfile(input_file):
            #     continue
            if input_file_path not in input_file_dic:
                input_file_dic[input_file_path] = [os.path.basename(input_file)]
            else:
                input_file_dic[input_file_path].append(os.path.basename(input_file))
        logger.info("开始写txt")
        # 可能是处理跨天文件数据
        for k, v in input_file_dic.items():
            input_file = os.path.join(self.output_data_path, f"inputfile_CWR_{file_label}.txt")

            with open(input_file, "w") as f:
                # 暂时替换成本地目录（后面要替换回去，用挂在的minio路径，也就是Java传进来的文件路径）*****************************
                file_name = ''
                file_num = 0
                put_dir = transfer_path(os.path.dirname(k))
                for file in v:
                    # 只有同时存在CWG, CWP两个格式文件时,才作为输入文件执行
                    # if os.path.isfile(os.path.join(f'{put_dir}/{os.path.basename(k)}', file.replace(".CWG", ".CWP"))):
                    if os.path.isfile(os.path.join(put_dir, os.path.basename(k), file.replace(".CWG", ".CWP"))) and \
                       os.path.getsize(os.path.join(put_dir, os.path.basename(k), file)) > 0 and \
                       os.path.getsize(os.path.join(put_dir, os.path.basename(k), file.replace(".CWG", ".CWP"))) > 0:
                        file_num += 1
                        file_name += "".join("{}\n{}\n".format(file.replace(".CWG", ".CWP"), file))
                if file_num == 0:
                    raise FileExistsError("文件大小为0，数据异常")

                s = f"\t{file_num}\n{Path(os.path.join(put_dir, os.path.basename(k)))}\\\n" + file_name
                f.write(s)
            logger.info("{}构建成功".format(input_file))

            cmd = f"{self.exe_path} {input_file} {output_path}"
            logger.info("开始执行算法程序{}".format(os.path.basename(self.exe_path)))
            print(cmd)
            # 执行exe
            os.system(cmd)
            os.remove(input_file)
        if len(os.listdir(output_path)) == 0:
            raise Exception("算法程序执行失败，未生成数据")
        else:
            # 生产文件输出路径
            self.output_files = glob.glob(os.path.join(output_path, '*1.dat'))
            self.output_files_sz = glob.glob(os.path.join(output_path, '*2.dat'))
            #  ---中间文件生成消息并发送---
            # temp_file_info = [{"tempFiles": []}]
            # for idx, file in enumerate(self.output_files):
            #     temp_ = temp_message([file], [self.fileid[idx]])
            #     temp_file_info[0]["tempFiles"] = temp_file_info[0].get("tempFiles", []) + [temp_[0]["tempFiles"][0]]
            temp_file_info = temp_message([self.output_files[0], self.output_files_sz[0]], self.fileid)
            publish_temp_msg("20", self.jobid, "CWR", temp_file_info)

    def select_data(self, data, start_time, end_time):
        tme_t = ~((np.array(data.time_list) < datetime.strptime(start_time, "%Y%m%d%H%M%S")) | (
                    np.array(data.time_list) > datetime.strptime(end_time, "%Y%m%d%H%M%S")))
        np.array(data.bright_belt_bottom_height)[tme_t].tolist()

    def get_data_and_draw(self, eff_time=None, hail_time=None):
        logger.info("开始读取数据...")
        cwr_data = CWRData(self.output_files, self.start_time, self.end_time)
        logger.info("开始绘图...")
        # elements = ("raw_ref", "raw_vel", "raw_sw", "qc_ref", "qc_vel", "qc_sw")
        elements = {"ref": ("raw_ref", "qc_ref"), "vel": ("raw_vel", "qc_vel"), "sw": ("raw_sw", "qc_sw"),
                    "cloud_boundary": ("raw_ref", "qc_ref"), "vair": ["vair"]}
        pic_list = []
        line_data = []
        if not cwr_data.time_list:
            raise Exception("无法匹配对应CWP文件。")  # 算法程序没有生成数据文件，请排查算法程序是否兼容数据!
        # 廓线
        element_all = ["ref", "vel", "sw", "cloud_boundary", "vair"] if self.element is None else [self.element]
        for self.element in element_all:
            if self.analysis_type == 1:
                time_idx = cwr_data.time_list.index(datetime.strptime(self.data_time, "%Y%m%d%H%M%S"))
                for ele in elements[self.element]:
                    data = np.array(eval("cwr_data.{}".format(ele)), dtype=float)
                    y = list(
                        range(cwr_data.lib_length[0], data.shape[1] * cwr_data.lib_length[0] + 1, cwr_data.lib_length[0]))
                    y = [y_km / 1000 for y_km in y]
                    data[data <= -999] = np.nan
                    c_profile_config[ele]["x"] = np.around(data, 2).tolist()[time_idx]
                    c_profile_config[ele]["y"] = [y]
                    c_profile_config[ele]["yname"] = [self.title]
                    c_profile_config[ele]["time"] = str(cwr_data.time_list[time_idx])
                    line_data.append(c_profile_config[ele])
                return pic_list, line_data

            x = cwr_data.time_list
            x_str = [t.strftime("%H:%M") for t in cwr_data.time_list]
            # 绘图
            for ele in elements[self.element]:
                data = np.array(eval("cwr_data.{}".format(ele)), dtype=float)
                y = list(
                    range(cwr_data.lib_length[0], data.shape[1] * cwr_data.lib_length[0] + 1, cwr_data.lib_length[0]))
                y = [y_km / 1000 for y_km in y]

                if data.shape[0] < 2:
                    raise Exception("当前时间筛选数据不足，无法绘制填色图")
                if "raw" in ele:
                    pic_path = os.path.join(self.pic_file_path, str(self.jobid), "old")
                else:
                    pic_path = os.path.join(self.pic_file_path, str(self.jobid), "new")
                if not os.path.isdir(pic_path):
                    os.makedirs(pic_path)
                pic_file = os.path.join(pic_path, "{}_{}_{}.png".format(ele, cwr_data.time_list[0].strftime("%Y%m%d%H%M%S"),
                                                                        cwr_data.time_list[-1].strftime("%Y%m%d%H%M%S")))
                # 增加空白时间数据
                # time_range1 = pd.date_range(x_time[0], x_time[-1], freq='s')
                # distribution_df = merge_data(time_range1, x_time, data)
                # data = np.array(distribution_df.iloc[:, 1:], dtype=float)
                # x = distribution_df.data_time.tolist()
                # x_str = [t.strftime("%H:%M") for t in x]
                # 存取云顶、云高绘图数据参数
                cloud_top = []
                cloud_bottom = []
                bright_belt_bottom_height = []
                bright_belt_top_height = []
                if self.element == 'cloud_boundary':
                    cloud_top = list_to_array(eval("cwr_data.{}".format('cloud_top')), np.nan, dtype=float)
                    cloud_bottom = list_to_array(eval("cwr_data.{}".format('cloud_bottom')), np.nan, dtype=float)
                    bright_belt_bottom_height = np.array(eval("cwr_data.{}".format('bright_belt_bottom_height')),
                                                         dtype=float) * 30 / 1000
                    bright_belt_top_height = np.array(eval("cwr_data.{}".format('bright_belt_top_height')),
                                                      dtype=float) * 30 / 1000
                    bright_belt_bottom_height[bright_belt_bottom_height < 0] = None
                    bright_belt_top_height[bright_belt_top_height < 0] = None

                    if cloud_top.any() and cloud_bottom.any():
                        cloud_top = cloud_top * 30 / 1000  # 数据转为km
                        cloud_bottom = cloud_bottom * 30 / 1000
                    pic_file = os.path.join(pic_path, "{}_{}_{}.png".format('cloud_boundary',
                                                                            cwr_data.time_list[0].strftime("%Y%m%d%H%M%S"),
                                                                            cwr_data.time_list[-1].strftime(
                                                                                "%Y%m%d%H%M%S")))

                pic_config = c_pic_config[ele]
                unit = pic_config["unit"]
                # x_label = pic_config["x_label"]
                x_label = cwr_data.time_list[0].strftime("%Y-%m-%d")
                y_label = pic_config["y_label"]
                levels = pic_config["vlim"]
                colors = pic_config["cmap"]
                titles = pic_config["title"].split("时间")[0]
                title_time = '~'.join(
                    [cwr_data.time_list[0].strftime("%Y-%m-%d %H:%M"), cwr_data.time_list[-1].strftime("%Y-%m-%d %H:%M")])
                # if ele.split('_')[0] == 'raw':
                #     pic_title = f'{self.title}质控前\t{title_time}'
                # else:
                #     pic_title = f'{self.title}\t{title_time}'
                pic_title = f'{self.title}\t{titles}\t{title_time}'
                if self.element == 'cloud_boundary':
                    pic_title = f'{self.title}\t{titles}(含云顶云底)\t{title_time}'
                draw_color_image(x, y, data, unit, x_label, y_label, pic_title, pic_file, levels, colors, x_str,
                                 ylims=self.ylims,
                                 cloud_top=cloud_top, cloud_bottom=cloud_bottom,
                                 bbbh=bright_belt_bottom_height, bbth=bright_belt_top_height,
                                 eff_time=eff_time, hail_time=hail_time)
                pic_list.append({
                    "element": self.element, "fileName": os.path.basename(pic_file),
                    "path": transfer_path(pic_file, is_win_path=True),
                    "img_type": "figure"
                })

                # 添加时间序图信息
                # if self.analysis_type == 5:
                if self.element == "cloud_boundary":
                    continue
                if self.height is None:
                    height_index = 0
                else:
                    height = self.height / 1000
                    if height < y[0] or height > y[-1]:
                        raise ValueError("高度请在{}之间选。".format([y[0] * 1000, y[-1] * 1000]))
                    height_index = y.index(height)
                c_line_config[ele]["x"] = [dt.strftime("%Y%m%d%H%M%S") for dt in cwr_data.time_list]
                y_data = data[:, height_index]
                y_data[y_data == -999] = np.nan
                y_data = np.around(y_data, 2)
                c_line_config[ele]["y"] = [y_data.tolist()]
                c_line_config[ele]["yname"] = [self.title + '\t' + pic_config["name"]]
                c_line_config[ele]["otherData"] = {"args": [{
                    "argsType": "height", "defaultValue": float(y[0] * 1000),
                    "specialArgs": np.around((np.array(y) * 1000), 0).tolist(), "unit": "m"}
                ]}
                line_data.append(c_line_config[ele])

        return pic_list, line_data

    def run(self, eff_time=None, hail_time=None):
        self.run_exe()
        if not self.output_files:
            raise ValueError("找不到数据数据文件，请排查输入文件是否存在")

        if not self.is_pic:
            cwr = CWRData_nc(self.output_files)
            output_ncdata_path = os.path.dirname(self.output_files[0]).replace(self.output_data_path,
                                                                               self.output_ncdata_path)
            outfile_qc = cwr.save_to_qc_nc(output_ncdata_path)
            outfile_raw = cwr.save_to_raw_nc(output_ncdata_path)
            #  ---中间文件生成消息并发送---
            temp_file_info = temp_message([outfile_qc, outfile_raw], self.fileid)
            publish_temp_msg("20", self.jobid, "CWR", temp_file_info)
            pic_list, line_data = [], []
            self.output_files.append(outfile_qc)
            self.output_files.append(outfile_raw)
        else:  # 读取数据并画图
            pic_list, line_data = self.get_data_and_draw(eff_time=eff_time, hail_time=hail_time)

        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, [self.output_files, self.output_files_sz])
        return [{"picFiles": pic_list, "picData": line_data}]


if __name__ == '__main__':
    input_filelist = [
        {
            "filePath": r"gzfb\code\GZFB\code\radar_analysis_pic\data\CWR\data\20220520\CWR20220520000054.CWG",
            "jobFileId": 1,
            "storeSource": "minio",
            "fileId": 3,
        },
        {
            "filePath": r"gzfb\code\GZFB\code\radar_analysis_pic\data\CWR\data\20220520\CWR20220520000057.CWG",
            "jobFileId": 1,
            "storeSource": "minio",
            "fileId": 3,
        },
        {
            "filePath": r"gzfb\code\GZFB\code\radar_analysis_pic\data\CWR\data\20220520\CWR20220520000101.CWG",
            "jobFileId": 1,
            "storeSource": "minio",
            "fileId": 3,
        },
        # {
        #     "filePath": r"gzfb\code\GZFB\code\radar_analysis_pic\data\CWR\data\20220520\CWR20220520000145.CWG",
        #     "jobFileId": 1,
        #     "storeSource": "minio",
        #     "fileId": 3,
        # }
    ]
    height = 30
    jobid = 2
    title = ""
    # element = 'vel'
    element = 'cloud_boundary'
    analysis_type = 1
    import time

    t = time.time()
    mrr = CWRController(input_filelist, height, jobid, element, title, analysis_type)
    mrr.run()
    print("耗时：", time.time() - t)
