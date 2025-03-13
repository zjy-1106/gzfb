# -*- coding: utf-8 -*-
import glob
import json
import os
from pathlib import Path

from metpy.units import units
from pandas.core.frame import DataFrame
import numpy as np
import datetime
import metpy.calc as mpcalc
from sklearn.metrics import mean_squared_error

from config.config import config_info
from service.utils import transfer_path, file_exit
# from utils.file_uploader import minio_client
BASE_PATH = os.path.dirname(os.path.dirname(__file__))


def select_data(h, tem, rh):
    h, tem, rh = np.array(h), np.array(tem), np.array(rh)
    res1, res2 = [], []
    for i in range(2000, 12500, 500):
        # 每隔500米取一个平均值
        select_h = ~(~(i < h) + (h > i + 500))
        select_h = np.broadcast_to(select_h, tem.shape)
        if len(tem.shape) == 1:
            res1.append(np.mean(tem[select_h]))
            res2.append(np.mean(rh[select_h]))
        else:
            res1.append(np.mean(tem[select_h].reshape(len(tem), -1), 1))
            res2.append(np.mean(rh[select_h].reshape(len(tem), -1), 1))

    return res1, res2


class MwrLliners:
    def __init__(self, input_filelist):
        self.file_names = input_filelist
        self.data_time = []
        self.temp_list = []
        self.rh_list = []
        self.height_list = []

    def get_mwr_data(self):
        # 返回折线图数据
        path_cfg = config_info.get_mwr_config
        output_data_path = transfer_path(path_cfg["exe_out_path"], is_win_path=True, is_inline_path=True)  # 输出路径
        out_dir = [glob.glob(os.path.join(output_data_path,
                                          time_str.strftime("%Y%m%d%H%M%S")[0:8],
                                          time_str.strftime("%Y%m%d%H%M%S")[8:10],
                                          time_str.strftime("%Y%m%d%H%M%S")[10:], "*.dat"))[0]
                   for time_str in self.data_time]
        for out_file in out_dir:
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
                    height_list = data[2].split()[2:]  # 83个高度层
                    if not self.height_list:  # 高度层数据只保留一份
                        self.height_list = [float(height) * 1000 + 2496.89 for height in height_list]
                    temp_list = data[3].split()[1:]  # 温度
                    self.temp_list.append(temp_list)
                    rh_list = data[4].split()[2:]  # 湿度
                    self.rh_list.append(rh_list)

    def get_l_data(self):
        l_h, l_t, l_rh = [], [], []

        for file_name in self.file_names:
            file_name = file_name.get('filePath')
            for dir_path, dir_names, file_names in os.walk(file_name):
                if file_names:
                    for file in file_names:
                        with open(os.path.join(dir_path, file), mode='rb') as f:
                            content_list = [i.strip() for i in f]
                        jsons = content_list[0].decode()
                        df = DataFrame(json.loads(jsons), dtype=float)
                        df[df >= 999998] = None
                        data = df.sort_values(by='PRS_HWC', ascending=False)
                        self.data_time.append(datetime.datetime(int(data.Year_Data[0]), int(data.Mon_Data[0]),
                                                                int(data.Day_Data[0]),
                                                                int(data.Hour_Data[0])) + datetime.timedelta(hours=8))

                        td = np.array(data['DPT'], dtype=float)
                        tem = np.array(data['TEM'], dtype=float)
                        e_td = [mpcalc.saturation_vapor_pressure(t_d * units.degC).m for t_d in td]
                        e_t = [mpcalc.saturation_vapor_pressure(t * units.degC).m for t in tem]
                        rh = np.array(e_td) / np.array(e_t) * 100

                        l_h.append(np.array(data['GPH'], dtype=float))
                        l_t.append(np.array(data['TEM'], dtype=float))
                        l_rh.append(rh)

        return l_h, l_t, l_rh

    def run(self):
        pic_info = []
        line_data = []

        l_h, l_t, l_rh = self.get_l_data()

        self.get_mwr_data()

        lt, lrh = [], []
        for k in range(len(l_h)):
            lt1, lrh1 = select_data(l_h[k], l_t[k], l_rh[k])
            lt.append(lt1)
            lrh.append(lrh1)
        lt = np.array(lt).T
        lrh = np.array(lrh).T

        mwr_h = np.array(self.height_list, dtype=float)
        mwr_t = np.array(self.temp_list, dtype=float)
        mwr_rh = np.array(self.rh_list, dtype=float)
        mwrt, mwrrh = select_data(mwr_h, mwr_t, mwr_rh)

        rmse_t = []
        rmse_rh = []
        for i in range(len(lt)):
            rmse_t.append(np.sqrt(mean_squared_error(lt[i], mwrt[i])))
            rmse_rh.append(np.sqrt(mean_squared_error(lrh[i], mwrrh[i])))
        data_t = np.array(self.temp_list, dtype='float32').T
        data_rh = np.array(self.rh_list, dtype='float32').T
        for j, i in enumerate(range(2000, 12500, 500)):
            a = np.argwhere(mwr_h > i)[:, 0]
            b = np.argwhere(mwr_h < i + 500)[:, 0]
            ids = list(set(a).intersection(set(b)))
            data_t[ids] = data_t[ids] - rmse_t[j]
            data_rh[ids] = data_rh[ids] - rmse_rh[j]

        path_cfg = config_info.get_mwr_config
        res_path = os.path.dirname(path_cfg["exe_path"])  # os.path.join(BASE_PATH, 'exe', 'MWR')
        if not Path(res_path).exists():
            Path.mkdir(Path(res_path), parents=True, exist_ok=True)
        filename = os.path.join(res_path, 'err.txt')
        with open(filename, 'w') as f:
            f.write('{' + f'"temperature": {rmse_t},')
            f.write(f'"RH": {rmse_rh}' + '}')
        # with open(filename, 'r') as f:
        #     rmse = json.loads(f.readlines()[0])

        return filename


if __name__ == '__main__':
    fi = [{'filePath': r'D:\gzfb_test\mwr_l\Lbr\LBR_20230810080000.txt'},
          {'filePath': r'D:\gzfb_test\mwr_l\Lbr\LBR_20230810080000.txt'},
          {'filePath': r'D:\gzfb_test\mwr_l\Lbr\LBR_20230810080000.txt'}]
    MwrLliners(fi).run()
