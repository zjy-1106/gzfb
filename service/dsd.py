# -*- coding: utf-8 -*-
"""
@Time : 2022/7/11 15:10
@Author : YangZhi
"""
import glob
import os
import re

from loguru import logger

import numpy as np
from config.config import config_info
from module.dsd.draw import DSD
from mq.pubulisher import publish_temp_msg
from service.dsd_radar import DsdRadar
from service.utils import transfer_path, temp_message, upload_files
from module.public.avaliable_product import dsd_image_type
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


class DsdController:
    def __init__(self, input_filelist, jobid, element, start_time, end_time, density_time=None):
        self.ori_file_list = sorted(input_filelist, key=lambda x: x.get("filePath"))    # 对文件名排序的文件信息
        self.jobid = jobid
        self.start_time = start_time
        self.end_time = end_time
        self.element = element  # 要素（echo/raininess/density/...）
        self.density_time = density_time
        path_cfg = config_info.get_dsd_config
        self.output_data_path = path_cfg["exe_out_path"]  # 输出路径
        self.pic_file_path = path_cfg["pic_path"] + '/' + str(jobid)  # 生成图片保存的路径
        self.exe_path = path_cfg["exe_path"]  # os.path.join(BASE_PATH, 'exe', 'DSD', 'DSD.exe')  exe路径
        self.data_path = path_cfg["origin_data"]  # 源数据路径
        self.scatter_data_path = os.path.join(os.path.dirname(self.exe_path), "Scatterdata")  # 散射数据

        self.data_file_list = []  # 算法生成数据文件
        self.pic_list = []  # 生成的图片
        self.pic_data = []

    def run_exe(self):
        if not os.path.isdir(self.output_data_path):
            os.makedirs(self.output_data_path)
        if not os.path.isdir(self.pic_file_path):
            os.makedirs(self.pic_file_path)

        for f in self.ori_file_list:
            # 20220614_R7237_YDP.TXT 雨滴谱文件名
            time_str = re.search(r'\d{6}', f.get('filePath').split('/')[-2]).group()
            out_dir = os.path.join(self.output_data_path, time_str, str(f.get('fileId')))
            input_file = transfer_path(f.get('filePath'))
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            try:
                out_file = os.listdir(transfer_path(out_dir, is_win_path=True, is_inline_path=True))
            except FileNotFoundError:
                out_file = []
            if out_file:
                out_file = os.path.join(transfer_path(out_dir, is_win_path=True, is_inline_path=True), out_file[0])
                logger.info("文件{}已存在，无需重复执行算法程序。".format(out_file))
            else:
                logger.info("开始解析：{}".format(input_file))
                if os.path.getsize(input_file) == 0:    # 部分文件大小为0，无数据
                    raise Exception("原始文件错误")
                # 执行exe
                cmd = f"{self.exe_path} {self.scatter_data_path} {input_file} {out_dir}"
                os.system(cmd)
                if not os.listdir(out_dir):
                    logger.error("dsd执行{}失败".format(input_file))
                    self.ori_file_list.pop(len(self.data_file_list))  # 删除该索引对应的源文件（和算法程序生成的文件一一对应）
                    continue
                else:
                    #  ---中间文件生成消息发送---
                    out_file = glob.glob(os.path.join(out_dir, "*.dat"))[0]
                    temp_file_info = temp_message([out_file], [f.get('fileId')])
                    publish_temp_msg("20", self.jobid, "DSD", temp_file_info)

            self.data_file_list.append(out_file)

    def get_file_info(self, img_file, file_time_list):
        # 图片信息数据
        pic_info = {
            "element": self.element, "fileName": os.path.basename(img_file),
            "path": transfer_path(img_file, is_win_path=True),
            "img_type": "figure"
        }
        if self.element == 'diameter_speed_density':
            time_list = [f"{t:%Y%m%d%H%M%S}" for t in file_time_list]
            pic_info.update({"otherData": {"dataTimeList": time_list}})
        return pic_info

    def get_data(self, data_file, ori_file_info, data_type, start_t, end_t, title, eff_time=None, hail_time=None):
        """
        读取文件中单独要素数据
        :param data_file: 数据文件
        :param ori_file: 源文件
        :param data_type: 数据种类（质控前/指控后）
        :param start_t:
        :param end_t:
        :param title:
        :return:
        """
        ori_file = []
        for i, or_f in enumerate(ori_file_info):
            ori_file.append(transfer_path(or_f.get('filePath')))
        # data_file = transfer_path(data_file)
        draw_obj = DSD()
        if data_type == "old":  # 质控前
            return draw_obj.QC_before(ori_file, "", data_file)
        elif data_type == "new":  # 质控后
            # 绘制色斑图
            if self.element in dsd_image_type:
                dsd = DsdRadar(start_t, end_t, self.element, data_file, title)
                img_path = os.path.join(self.pic_file_path, "new")
                img_file, file_time_list = dsd.draw(img_path, title, True, self.density_time,
                                                    eff_time=eff_time, hail_time=hail_time)
                self.pic_list.append(self.get_file_info(img_file, file_time_list))

                img_path = os.path.join(self.pic_file_path, "old")
                img_file, file_time_list = dsd.draw(img_path, title+'质控前', False, self.density_time,
                                                    eff_time=eff_time, hail_time=hail_time)
                self.pic_list.append(self.get_file_info(img_file, file_time_list))
            return draw_obj.QC_after(data_file, "")
        else:
            raise Exception("数据类型输入有误")

    def draw(self, data_list, element, is_qc=False):
        draw_obj = DSD()
        if is_qc:   # 质控前
            img_map = {"raininess": draw_obj.draw_yq_zkq,
                       "echo": draw_obj.draw_hb_ys}
        else:
            img_map = {"raininess": draw_obj.draw_yq,
                       "water": draw_obj.draw_hsl,
                       "echo": draw_obj.draw_hb,
                       "ZDR": draw_obj.draw_ZDR,
                       "KDP": draw_obj.draw_KDP,
                       "Lattenuation": draw_obj.draw_sppz,
                       "Vattenuation": draw_obj.draw_czpz}

        return img_map[element](data_list)

    def run(self, title, eff_time=None, hail_time=None):
        dsd_zkq_list = ["raininess", "echo"]  # 雨滴谱质控前的图片种类
        dsd_zkh_list = ["raininess", "water", "echo", "ZDR", "KDP", "Lattenuation", "Vattenuation"]  # 微雨雷达质控后的图片种类
        self.run_exe()
        for isqc in ("old", "new"):
            data_list = []
            # 便利读取数据
            # for i in range(len(self.data_file_list)):
            data_list = self.get_data(self.data_file_list, self.ori_file_list, isqc,
                                      self.start_time, self.end_time, title, eff_time=eff_time, hail_time=hail_time)
                # data_list += data
        if not data_list:
            raise Exception(f"{title}:文件中没有对应时间段数据")
        if isqc == "old":
            if self.element in dsd_zkq_list:
                x, y, x_label, y_label = self.draw(data_list, self.element, True)
                y = np.around(y, 2)
                x_filter = []
                y_filter = []
                if self.start_time and self.end_time:
                    for idx, t in enumerate(x):
                        if self.start_time < t < self.end_time:
                            x_filter.append(t)
                            y_filter.append(y[0][idx])
                else:
                    x_filter = x.tolist()
                    y_filter = y.squeeze()
                    y_filter = y_filter.tolist()
                self.pic_data.append({
                    "element": self.element, "x": x_filter, "y": [y_filter], "xlabel": x_label,
                    "ylabel": y_label, "yname": title, "picType": 0
                })
        else:
            if self.element in dsd_zkh_list:
                x, y, x_label, y_label = self.draw(data_list, self.element)
                y = np.around(y, 2)
                x_filter = []
                y_filter = []
                if self.start_time and self.end_time:
                    for idx, t in enumerate(x):
                        if self.start_time < t < self.end_time:
                            x_filter.append(t)
                            y_filter.append(y[0][idx])
                else:
                    x_filter = x.tolist()
                    y_filter = y.squeeze()
                    y_filter = y_filter.tolist()
                self.pic_data.append({
                    "element": self.element, "x": x_filter, "y": [y_filter], "xlabel": x_label,
                    "ylabel": y_label, "yname": title, "picType": 1
                })
        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, self.data_file_list)
        return [{"picFiles": self.pic_list, "picData": self.pic_data}]
