# -*- coding: utf-8 -*-
"""
@Time : 2022/7/11 14:09
@Author : YangZhi
"""
import glob
import os
import re


from loguru import logger
import numpy as np
from config.config import config_info
from module.mrr.draw import MRR
from mq.pubulisher import publish_temp_msg
from service.utils import transfer_path, temp_message, upload_files
from service.mrr_radar_ori import MRROri
from service.mrr_radar import MRRRadar
from module.public.avaliable_product import mrr_image_type_ori, mrr_image_type

"""
微雨雷达算法，并出图
"""
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


class MrrController:
    def __init__(self, input_filelist, height, jobid, element, start_time, end_time):
        self.ori_file_list = sorted(input_filelist, key=lambda x: x.get("filePath"))  # 对文件名排序的文件信息

        # 过滤时间段数据
        self.start_time = start_time
        self.end_time = end_time
        self.element = element  # 要素（echo/raininess/density/...）
        self.height = height if height else 100  # 高度层
        self.jobid = jobid
        height_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
                       1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100]
        try:
            # self.height_index = height_list.index(self.height)
            self.height_index = np.argmin(abs(np.array(height_list) - self.height))
        except ValueError:
            raise Exception(f"高度层参数错误, {self.height}")

        path_cfg = config_info.get_mrr_config
        self.output_data_path = path_cfg["exe_out_path"]  # 输出路径
        self.pic_file_path = path_cfg["pic_path"] + '/' + str(jobid)  # 生成图片保存的路径
        self.exe_path = path_cfg["exe_path"]  # os.path.join(BASE_PATH, 'exe', 'MRR', 'MRR.exe')  # exe路径
        self.scatter_data_path = os.path.join(os.path.dirname(self.exe_path), "Scatterdata")  # 散射数据

        self.data_file_list = []  # 算法生成数据文件
        self.pic_list = []  # 生成的图片
        self.pic_data = []

    def run_exe(self):
        if not os.path.isdir(self.output_data_path):
            os.makedirs(self.output_data_path)
        if not os.path.isdir(self.pic_file_path):
            os.makedirs(self.pic_file_path)
        # 执行exe程序
        for f in self.ori_file_list:
            time_str = re.search(r'\d{6}', f.get('filePath').split('/')[-2]).group()
            out_dir = os.path.join(self.output_data_path, time_str, str(f.get('fileId')))

            input_file = transfer_path(f.get('filePath'))
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            out_file = glob.glob(os.path.join(transfer_path(out_dir, is_win_path=True, is_inline_path=True), "*mrr.dat"))
            if out_file:
                out_file = os.path.join(transfer_path(out_dir, is_win_path=True, is_inline_path=True), out_file[0])
                logger.info("文件{}已存在，无需重复执行算法程序。".format(out_file))

            else:
                out_dir = os.path.join(out_dir, str(self.jobid))
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                logger.info("开始解析：{}".format(input_file))
                # 执行exe
                cmd = f"{self.exe_path} {self.scatter_data_path} {input_file} {out_dir}"
                os.system(cmd)
                print(cmd)
                if not os.listdir(out_dir):
                    logger.error("mrr执行{}失败".format(input_file))
                    self.ori_file_list.pop(len(self.data_file_list))  # 删除该索引对应的源文件（和算法程序生成的文件一一对应）
                    continue
                else:
                    out_file = glob.glob(os.path.join(out_dir, "*"))[0]
                    temp_file_info = temp_message([out_file], [f.get('fileId')], self.jobid)
                    publish_temp_msg("20", self.jobid, "MRR", temp_file_info)

            self.data_file_list.append(out_file)

    def draw_color_pic(self, title, ylims=None, eff_time=None, hail_time=None):
        # 绘制色斑图-指控前
        if self.element in mrr_image_type_ori:
            mrr = MRROri(self.start_time, self.end_time, self.element, self.height_index,
                         os.path.join(self.pic_file_path, "old"), self.ori_file_list, title)
            img_file = mrr.draw(title + '质控前', ylims, eff_time=eff_time, hail_time=hail_time)
            self.pic_list.append({
                "element": self.element, "fileName": os.path.basename(img_file),
                "path": transfer_path(img_file, is_win_path=True),
                "img_type": "figure"
            })
        # 质控后
        if self.element in mrr_image_type:
            mrr = MRRRadar(self.start_time, self.end_time, self.element, title,
                           os.path.join(self.pic_file_path, "new"), self.data_file_list, self.height_index)
            img_file = mrr.draw(title, ylims, eff_time=eff_time, hail_time=hail_time)
            self.pic_list.append({
                "element": self.element, "fileName": os.path.basename(img_file), "data_type": "new",
                "path": transfer_path(img_file, is_win_path=True),
                "img_type": "figure"
            })

    def get_data(self, data_file, ori_file, data_type, start_t, end_t, title):
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
        draw_obj = MRR()
        if data_type == "old":  # 质控前
            ori_file = transfer_path(ori_file)
            return draw_obj.QC_before(ori_file, "")
        elif data_type == "new":  # 质控后
            data_file = transfer_path(data_file)
            return draw_obj.QC_after(data_file, "")
        else:
            raise Exception("数据类型输入有误")

    def draw(self, data_list, element):
        draw_obj = MRR()
        img_map = {"raininess": draw_obj.draw_yq,
                   "water": draw_obj.draw_hsl,
                   "echo": draw_obj.draw_hb_sj,
                   "echos": draw_obj.draw_hb,
                   "ZDR": draw_obj.draw_ZDR,
                   "density": draw_obj.draw_nw,
                   "diameter": draw_obj.draw_dm,
                   "Mu": draw_obj.draw_mu
                   }

        return img_map[element](data_list, self.height_index)

    def get_data_by_is_qc(self, title, is_qc):
        data_list = []
        # 便利读取数据
        for i in range(len(self.data_file_list)):
            data = self.get_data(self.data_file_list[i], self.ori_file_list[i].get('filePath'), is_qc,
                                 self.start_time, self.end_time, title)
            if data:
                data_list += data
        if not data_list:
            raise Exception(f"{title}:文件中没有对应时间段数据")

        return data_list

    def run(self, title, ylims=None, eff_time=None, hail_time=None):
        mrr_zkq_list = ["raininess", "water", "echo"]  # 微雨雷达质控前的图片种类
        mrr_zkh_list = ["raininess", "water", "echo", "echos", "ZDR", "density", "diameter", "Mu"]  # 微雨雷达质控后的图片种类
        self.run_exe()

        # 绘制色斑图
        self.draw_color_pic(title, ylims, eff_time=eff_time, hail_time=hail_time)

        # 质控前
        if self.element in mrr_zkq_list:
            # 获取质控前数据
            data_list = self.get_data_by_is_qc(title, is_qc='old')
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
                "ylabel": y_label, "yname": title, "picType": 0,
                "otherData": {
                    "args": [{
                        "argsType": "height", "defaultValue": float(data_list[0]['02'][0]),
                        "specialArgs": np.array(data_list[0]['02'], dtype=float).tolist(), "unit": "m"}
                    ]}
            })
        # 质控后
        if self.element in mrr_zkh_list:
            # 获取质控后数据
            data_list = self.get_data_by_is_qc(title, is_qc='new')
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
                "ylabel": y_label, "yname": title, "picType": 1,
                "otherData": {
                    "args": [{
                        "argsType": "height", "defaultValue": float(data_list[0]['02'][0]),
                        "specialArgs": np.array(data_list[0]['02'], dtype=float).tolist(), "unit": "m"}
                    ]}
            })
        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, self.data_file_list, self.jobid)
        return [{"picFiles": self.pic_list, "picData": self.pic_data}]
