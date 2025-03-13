# -*- coding: utf-8 -*-
import glob
import json
import os
import re
import shutil
import time


import matplotlib
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
from loguru import logger
from matplotlib.dates import DateFormatter, rrulewrapper, RRuleLocator
import zipfile

from mq.pubulisher import publish_temp_msg
from service.utils import transfer_path, temp_message, ax_add_time_range
from config.config import config_info, cr_cwr_pic_config
from service.cr_cwr_deviation import Deviation
from service.effect_analysis import set_x_time
# from utils.file_uploader import minio_client

matplotlib.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


def unzip_file(file, out_path):
    """
    解压文件
    param file: 原压缩文件
    param out_path: 解压路径
    return: 解压后的文件位置
    """
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    f = zipfile.ZipFile(file, 'r')  # 压缩文件位置
    real_file = f.extract(f.namelist()[0], out_path)  # 解压位置
    f.close()
    return real_file


def get_data_call_service_api(payload):
    payload = json.dumps(payload)
    # service_name = 'cl-arithmetic-service'
    # endpoint = '/arithmetic/getFusionCloudRainFileResp'
    # headers = {'Content-Type': 'application/json'}
    # service_instances_url = f'http://nacos.clizard.team:8848/nacos/v1/ns/instance/list?serviceName={service_name}&&namespaceId=gzfb-project&&groupName=uat'
    # response = requests.get(service_instances_url)
    # instances = response.json()['hosts']
    # if not instances:
    #     raise Exception(f"No instances available for service: {service_name}")

    # instance = instances[0]  # 使用第一个实例，你可以根据需求进行选择
    server_info = config_info.get_data_server_cfg
    logger.info(f"===>{payload}")
    service_url = server_info['data_service_url']
    headers = server_info['headers']
    response = requests.post(service_url, data=payload, headers=headers)

    return response.json()


class CrCwr(object):

    def __init__(self, jobid, start_time, end_time):
        self.jobid = jobid
        get_data_meg = {"startTime": start_time,
                        "endTime": end_time}
        """获取需要的时间段的数据"""
        self.datafile = get_data_call_service_api(get_data_meg)
        for fi in self.datafile:
            if len(self.datafile[fi]) == 0:
                raise FileExistsError(f"{fi} 数据缺失")
        self.start_time = start_time
        self.end_time = end_time
        path_cfg = config_info.get_crcwr_cfg
        if start_time > '202402060000':  # 2024-02-06时C波段数据更改 库数：500到800，谱点数：512到256
            self.exe_path = path_cfg['exe_path_new']  # os.path.join(BASE_PATH, 'exe', 'CR_CWR', 'Prog2024.exe')
        else:
            self.exe_path = path_cfg['exe_path']  # os.path.join(BASE_PATH, 'exe', 'CR_CWR', 'Prog.exe')
        self.input_path = path_cfg['input_path']
        temp_dir = os.path.join(path_cfg['temp_path'], str(self.jobid))
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        self.unzip_temp_path = temp_dir
        self.count_file = os.path.join(self.input_path, 'Err.txt')
        self.output_path = os.path.join(path_cfg['output_path'], f'{start_time}_{end_time}')
        self.count_path_sca = os.path.dirname(self.exe_path)  # SCA常量文件
        self.pic_path = os.path.join(path_cfg['pic_path'], str(self.jobid))

        if not os.path.isdir(self.input_path):
            os.makedirs(self.input_path)
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

    def w_file(self, filename, unzip_path=None):
        start_time_hour = datetime.strptime(self.start_time, '%Y%m%d%H%M%S')
        end_time_hour = datetime.strptime(self.end_time, '%Y%m%d%H%M%S')
        file_path = os.path.join(self.input_path, f"inputfile_{filename}.txt")
        with open(file_path, "w") as f:
            f_s = ''
            file_num = 0 if unzip_path else 9846
            for file_name in self.datafile[filename]:
                file_name = transfer_path(file_name['filePath'])
                if unzip_path:
                    file_name = unzip_file(file_name, unzip_path)
                curr_time = datetime.strptime(re.search(r'\d{14}', file_name).group(), '%Y%m%d%H%M%S')
                if curr_time < start_time_hour:
                    # 只获取指定时间数据
                    continue
                if curr_time > end_time_hour:
                    break
                file_num += 1
                f_s += f"".join(f"{file_num} {file_name}\n")

            f.write(f_s.replace('/', '\\'))
        return file_path

    def create_input_file(self):
        input_file_fft = self.w_file('crFftFiles', self.unzip_temp_path)
        input_file_cwr_p = self.w_file('cwpFiles')
        input_file_cwr_g = self.w_file('cwgFiles')
        input_file_raw = self.w_file('crThiFiles', self.unzip_temp_path)

        with open(self.count_file, 'w') as cf:
            err_cr, err_cwr = Deviation(self.datafile, self.start_time, self.end_time).calculate()
            # err_cr, err_cwr = 0, 0
            cf.write(f'{err_cr}, {err_cwr}')
        logger.info("文件构建成功")

        return input_file_raw, input_file_fft, input_file_cwr_p, input_file_cwr_g

    def run_exe(self, raw_file, fft_file, c_cwp_file, c_cwg_file):
        start_time = time.time()
        cmd = f"{self.exe_path} {raw_file} {fft_file} {c_cwp_file} {c_cwg_file} {self.count_file} {self.count_path_sca} {self.output_path}"
        print(cmd)
        os.system(cmd)
        logger.info("执行exe耗时{}".format(time.time() - start_time))

    def draw(self, nc_file, ele, ylims=None, eff_time=None, hail_time=None):
        if not os.path.isdir(self.pic_path):
            os.makedirs(self.pic_path)
        title = ele
        with nc.Dataset(nc_file) as ds:
            data = ds.variables[ele][:]
            data = np.array(data)
            data[data == -999] = np.nan
            data[data >= 99999999] = np.nan
            str_s = ds.getncattr('Starting_Time(Seconds)')
            end_s = ds.getncattr('End_Time(Seconds)')
            units = ds.variables[ele].Units
        heights = np.linspace(30, data.shape[1] * 30, data.shape[1]) / 1000
        time_code = np.linspace(int(str_s), int(end_s), data.shape[0])
        times = [datetime.fromtimestamp(ti) for ti in time_code]
        heights, times = np.meshgrid(heights, times)

        if ylims is None:
            ylims = [np.nanmin(heights), np.nanmax(heights)]

        fig, ax = plt.subplots(figsize=(8, 3))

        if 'CR_Ref' in ele or 'LWC' in ele:
            levels = cr_cwr_pic_config[ele]["levels"]
            colors = cr_cwr_pic_config[ele]["colors"]
            # units = cr_cwr_pic_config[ele]["unit"]
            im = ax.contourf(times, heights, data, levels=levels, colors=colors)
        else:
            try:
                levels = cr_cwr_pic_config[ele]["vlim"]
                colors = cr_cwr_pic_config[ele]["cmap"]
                # units = cr_cwr_pic_config[ele]["unit"]
            except KeyError:
                levels = [None, None]
                colors = 'rainbow'
                # units = None

            im = ax.pcolormesh(times, heights, data, vmin=levels[0], vmax=levels[-1], cmap=colors, shading='auto')
            # im = ax.contourf(times, heights, data, cmap='rainbow')
            levels = None

        plt.xlabel(times[0][0].strftime('%Y-%m-%d'), fontsize=10)
        plt.ylabel('高度(km)', fontsize=10)
        plt.ylim(ylims)
        plt.title(title, fontsize=10)
        # position = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # (左，下，宽，高)
        br = plt.colorbar(im, fraction=0.033, pad=0.02, aspect=30, ticks=levels)  # , ticks=levels
        ax.tick_params(labelsize=8)
        br.ax.set_title(units, fontsize=12)
        br.ax.tick_params(labelsize=10)
        ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
        ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

        kwargs_major, kwargs_minor, timestyle = set_x_time(times[:, 0])
        majorformatter = DateFormatter(timestyle)
        rule1 = rrulewrapper(**kwargs_major)
        loc1 = RRuleLocator(rule1)
        ax.xaxis.set_major_locator(loc1)
        ax.xaxis.set_major_formatter(majorformatter)
        rule = rrulewrapper(**kwargs_minor)
        loc = RRuleLocator(rule)
        ax.xaxis.set_minor_locator(loc)
        # fig.autofmt_xdate()

        png_path = os.path.join(self.pic_path,
                                f'FusionCloudRain_{ele}_{times[0, 0].strftime("%Y%m%d%H%M%S")}_{times[-1, 0].strftime("%Y%m%d%H%M%S")}.png')
        plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        # 上传图片
        # # minio_client.put_object(png_path)

        return png_path

    def run(self, eff_time=None, hail_time=None):
        pic_list = []
        out_file = glob.glob(os.path.join(self.output_path, '*.NC'))
        inlune_out_file = glob.glob(os.path.join(transfer_path(self.output_path, is_win_path=True, is_inline_path=True), '*.NC'))
        if out_file:
            out_file = out_file[0]
            logger.info("文件{}已存在，无需重复执行算法程序。".format(out_file))
        elif inlune_out_file:
            out_file = inlune_out_file[0]
            logger.info("文件{}已存在，无需重复执行算法程序。".format(out_file))
        else:
            input_file_raw, input_file_fft, input_file_cwr_p, input_file_cwr_g = self.create_input_file()
            self.run_exe(input_file_raw, input_file_fft, input_file_cwr_p, input_file_cwr_g)
            shutil.rmtree(self.unzip_temp_path)  # 删除解压后的文件
            out_file = glob.glob(os.path.join(self.output_path, '*.NC'))[0]
            temp_file_info = temp_message(out_file, self.datafile['crThiFiles'][0]['fileId'])
            publish_temp_msg("20", self.jobid, "FusionCloudRain", temp_file_info)
        for ele in ['CR_Ref', 'CWR_Ref', 'RainfallRate', 'LWC', 'Vair', 'N0', 'Dm', 'Mu', 'f']:
            pic_file = self.draw(out_file, ele, eff_time=eff_time, hail_time=hail_time)
            pic_list.append({
                "element": ele, "fileName": os.path.basename(pic_file),
                "path": transfer_path(pic_file, is_win_path=True),
                "img_type": "figure"
            })

        # shutil.rmtree(self.unzip_temp_path)  # 删除解压后的文件

        # executor = ThreadPoolExecutor(max_workers=5)
        # executor.submit(upload_files, out_file)

        return [{"picFiles": pic_list}]


if __name__ == "__main__":
    fi = [{"filePath": r"E:\new_data\R7237\CDYWKA2\20230703\RAW_HH0703\Z_RADA_I_ST001_20230703090000_O_YCCR_YWKAAA_RAW_HH.BIN.zip"}]
    # cc = CrCwr(fi)
    # input_file_raw, input_file_fft, input_file_cwr_p, input_file_cwr_g = cc.create_input_file()
    # cc.run_exe(input_file_raw, input_file_fft, input_file_cwr_p, input_file_cwr_g)
