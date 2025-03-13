# -*- coding: utf-8 -*-
import glob
import os
import re
import time
import xarray as xr
import matplotlib
import numpy as np
import netCDF4 as nc
from datetime import datetime
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

"""
C波段连续波雷达算法程序
"""

matplotlib.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False


def list_to_array(x, fill_value=-999, dtype=int, default=1):
    """
    不规则list转array
    :param x: 列表
    :param fill_value: 空值填充值
    :param dtype: array数据类型
    :return:
    """
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(x)], axis=1)
    x = dff.fillna(fill_value).values.T.astype(dtype) * default
    x[x <= -999] = -99900
    return x


def list_to_array_1d(x, dtype=float, default=1):
    """
    规则list转array
    :param x: 列表
    :param dtype: array数据类型
    :param default:
    :return:
    """
    x = np.array(x, dtype=dtype) * default
    x[x <= -999] = -99900
    return x


class CWRData_nc:
    """读取C波段雷达算法程序生成的文件"""

    def __init__(self, data_files):
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
                            self.time_list.append(datetime.strptime(line.split(":")[1][:14], "%Y%m%d%H%M%S"))
                            # self.time_list.append(datetime.strptime(os.path.basename(file)[:14], "%Y%m%d%H%M%S"))
                            continue
                        elif "Bin num and Gatewidth(m):" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.lib_num.append(int(float(data[-2])))
                            self.lib_length.append(int(float(data[-1])))
                            continue
                        elif "Raw ref" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.raw_ref.append(data[2:])
                            continue
                        elif "Raw vel" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.raw_vel.append(data[2:])
                            continue
                        elif "Raw  sw" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.raw_sw.append(data[2:])
                            continue
                        elif "QC  ref" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.qc_ref.append(data[2:])
                            continue
                        elif "QC  vel" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.qc_vel.append(data[2:])
                            continue
                        elif "QC   sw" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.qc_sw.append(data[2:])
                            continue
                        elif "Vair" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.vair.append(data[1:])
                            continue
                        elif "Cloud layer and cloud boundary" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.cloud_layer.append(data[5])
                            # 云层为0时，没有云底和云顶
                            if int(data[5]) == 0:
                                self.cloud_bottom.append([-999])
                                self.cloud_top.append([-999])
                            else:
                                cloud_bottom_t = []
                                cloud_top_t = []
                                for i in range(int(data[5])):
                                    cloud_bottom_t.append(data[6 + i * 2])
                                    cloud_top_t.append(data[7 + i * 2])
                                self.cloud_bottom.append(cloud_bottom_t)
                                self.cloud_top.append(cloud_top_t)
                            continue
                        elif "Bright band height" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            self.bright_belt_bottom_height.append(data[4])
                            self.bright_belt_middle_height.append(data[5])
                            self.bright_belt_top_height.append(data[6])
                            continue
                        elif "QC SZ" in line:
                            data = [i for i in line.strip("\n").split(" ") if i != ""]
                            qc_sz_distance_library_number_tmp.append(data[2])
                            data_num = int(float(data[3]))  # 回波强度谱密度有效数据的个数
                            data_temp = data[4:4 + data_num]
                            qc_sz_radial_velocity_tmp.append(data_temp[::2])
                            qc_sz_wide_spectral_density_tmp.append(data_temp[1::2])
                            continue

            self.qc_sz_distance_library_number.append(qc_sz_distance_library_number_tmp)
            self.qc_sz_radial_velocity.append(qc_sz_radial_velocity_tmp)
            self.qc_sz_wide_spectral_density.append(qc_sz_wide_spectral_density_tmp)

    def save_to_qc_nc(self, outfile_path):
        # 保存数据
        # CFMCW_20230721_BASE_QC.nc
        outfile_path = os.path.join(outfile_path, f'{self.time_list[0].strftime("%Y%m%d")}')
        if not os.path.isdir(outfile_path):
            os.makedirs(outfile_path)
        outfile = os.path.join(outfile_path, f'CFMCW_{self.time_list[0].strftime("%Y%m%d")}_BASE_QC.nc')
        nc_obj = nc.Dataset(outfile, 'w', format='NETCDF4')

        nc_obj.createDimension('Bin_num', size=self.lib_num[0])
        nc_obj.createDimension('times', size=len(self.time_list))
        nc_obj.createDimension('gatewidth', size=self.lib_length[0])
        nc_obj.createDimension('cloud_layer_max', size=np.nanmax(np.array(self.cloud_layer, dtype=int)))

        y = nc_obj.createVariable('Bins', 'i4', 'Bin_num')
        times = nc_obj.createVariable('time', 'S19', 'times')
        sw = nc_obj.createVariable('qc_sw', 'i4', ('times', 'Bin_num'))
        ref = nc_obj.createVariable('qc_ref', 'i4', ('times', 'Bin_num'))
        vel = nc_obj.createVariable('qc_vel', 'i4', ('times', 'Bin_num'))
        vair = nc_obj.createVariable('vair', 'i4', ('times', 'Bin_num'))
        cloud_layer = nc_obj.createVariable('cloud_layer', 'i4', 'times')
        cloud_bottom = nc_obj.createVariable('cloud_bottom', 'i4', ('times', 'cloud_layer_max'))
        cloud_top = nc_obj.createVariable('cloud_top', 'i4', ('times', 'cloud_layer_max'))
        bright_band_bottom_height = nc_obj.createVariable('bright_band_bottom_height', 'i4', 'times')
        bright_band_middle_height = nc_obj.createVariable('bright_band_middle_height', 'i4', 'times')
        bright_band_top_height = nc_obj.createVariable('bright_band_top_height', 'i4', 'times')

        tie = [ti.strftime('%Y-%m-%d %H:%M:%S') for ti in self.time_list]
        y_l = np.array(np.linspace(1, self.lib_num[0], 500, dtype=int)) * self.lib_length[0]
        times[:] = np.array(tie)
        y[:] = np.array(y_l)
        sw[:] = list_to_array_1d(self.qc_sw, dtype=float, default=100)
        ref[:] = list_to_array_1d(self.qc_ref, dtype=float, default=100)
        vel[:] = list_to_array_1d(self.qc_vel, dtype=float, default=100)
        vair[:] = list_to_array_1d(self.vair, dtype=float, default=100)
        cloud_layer[:] = np.array(self.cloud_layer, dtype=int)  # 云层数
        cloud_bottom[:] = list_to_array(self.cloud_bottom, fill_value=-999, dtype=int, default=self.lib_length[0]) # 云底
        cloud_top[:] = list_to_array(self.cloud_top, fill_value=-999, dtype=int, default=self.lib_length[0])  # 云顶
        bright_band_bottom_height[:] = list_to_array_1d(self.bright_belt_bottom_height, dtype=float, default=self.lib_length[0])  # 零度层亮带的底高
        bright_band_middle_height[:] = list_to_array_1d(self.bright_belt_middle_height, dtype=float, default=self.lib_length[0])  # 零度层亮带的中间高
        bright_band_top_height[:] = list_to_array_1d(self.bright_belt_top_height, dtype=float, default=self.lib_length[0])  # 零度层亮带的顶高

        times.description = 'time, unlimited dimension'
        times.units = 'times %Y-%m-%d %H:%M:%S'  # format(datetime.timestamp(self.time_list[0]))
        y.description = 'range bins, unlimited dimension'
        y.units = 'm'
        sw.description = 'quality control spectrum width'
        sw.units = 'm/s * 100'
        ref.description = 'quality control reflectivity'
        ref.units = 'dBZ * 100'
        vel.description = 'quality control velocity'
        vel.units = 'm/s * 100'
        vair.description = 'vertical air motion'
        vair.units = 'm/s * 100'
        bright_band_bottom_height.description = 'bright band bottom height'
        bright_band_bottom_height.units = 'm'
        bright_band_middle_height.description = 'bright band middle height'
        bright_band_middle_height.units = 'm'
        bright_band_top_height.description = 'bright band top height'
        bright_band_top_height.units = 'm'
        cloud_bottom.description = 'cloud bottom height'
        cloud_bottom.units = 'm'
        cloud_top.description = 'cloud top height'
        cloud_top.units = 'm'
        cloud_layer.description = 'cloud layer'
        cloud_layer.units = ' '

        nc_obj.close()
        return outfile

    def save_to_raw_nc(self, outfile_path):
        # 保存数据
        # CFMCW_20230721_BASE_RAW.nc
        outfile_path = os.path.join(outfile_path, f'{self.time_list[0].strftime("%Y%m%d")}')
        if not os.path.isdir(outfile_path):
            os.makedirs(outfile_path)
        outfile = os.path.join(outfile_path, f'CFMCW_{self.time_list[0].strftime("%Y%m%d")}_BASE_RAW.nc')
        nc_obj = nc.Dataset(outfile, 'w', format='NETCDF4')

        nc_obj.createDimension('Bin_num', size=self.lib_num[0])
        nc_obj.createDimension('times', size=len(self.time_list))
        nc_obj.createDimension('gatewidth', size=self.lib_length[0])

        y = nc_obj.createVariable('Bins', 'i4', 'Bin_num')
        times = nc_obj.createVariable('time', 'S19', 'times')
        # gate_width = nc_obj.createVariable('gate', 'f8', 'gatewidth')
        sw = nc_obj.createVariable('raw_sw', 'i4', ('times', 'Bin_num'))
        ref = nc_obj.createVariable('raw_ref', 'i4', ('times', 'Bin_num'))
        vel = nc_obj.createVariable('raw_vel', 'i4', ('times', 'Bin_num'))

        # nc_obj.save_dimensions(y, 'y', dimensions_data_type='f8', dimensions_desc='y')
        tie = [ti.strftime('%Y-%m-%d %H:%M:%S') for ti in self.time_list]
        y_l = np.array(np.linspace(1, self.lib_num[0], 500, dtype=int)) * self.lib_length[0]
        # gate_width[:] = self.lib_length
        times[:] = np.array(tie)
        y[:] = y_l
        sw[:] = list_to_array_1d(self.raw_sw, dtype=float, default=100)
        ref[:] = list_to_array_1d(self.raw_ref, dtype=float, default=100)
        vel[:] = list_to_array_1d(self.raw_vel, dtype=float, default=100)

        times.description = 'time, unlimited dimension'
        times.units = 'times %Y-%m-%d %H:%M:%S'
        y.description = 'range bins, unlimited dimension'
        y.units = 'm'
        sw.description = 'quality control spectrum width'
        sw.units = 'm/s * 100'
        ref.description = 'quality control reflectivity'
        ref.units = 'dBZ * 100'
        vel.description = 'quality control velocity'
        vel.units = 'm/s * 100'

        nc_obj.close()
        return outfile


if __name__ == '__main__':
    cwr = CWRData_nc([r'F:\gzfb\data\20230721200002CWR_prod1.dat'])
    outfile_path1 = r'F:\gzfb\data\2023072'
    cwr.save_to_qc_nc(outfile_path1)
    cwr.save_to_raw_nc(outfile_path1)
    import netCDF4 as nc

    # f = r'D:\CFMCW_20230721_BASE_QC.nc'
    # with nc.Dataset(f, "r") as ds:
    #     data_ele = ds.variables['Bins'][:]
    #     np.array(ds.variables['yime'])


    # import os
    #
    # file_new = []
    # for i in range(5):
    #     fi = os.listdir(r'E:\data\FFT_MM\fft_nc')
    #     file = xr.open_dataset(os.path.join(r'E:\data\FFT_MM\fft_nc', fi[i]))['SZQC1']
    #     file_new.append(file)
    # da = xr.concat(file_new, dim='IPPInum')
    # da.to_netcdf('D:/all_file2.nc')

