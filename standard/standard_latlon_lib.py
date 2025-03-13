'''
standard_latlon_lib
=====

Provides
    1.本模块为标称转等经纬程序
    2.适用范围为FY4A L1及各类L2产品
    3.代码经过高度集成优化,目前可满足时效要求
    4.如发现BUG,请联系Author:Youxiaogang

How to use the model
----------------------------
    1.需要获取标称文件及所需数据集,一般为二维数组
    2.读取FY4A全圆盘经纬度静态文件,选取与投影文件相同的行列
    3.实例化StandardToLatLon类,传入参数,具体见类初始化函数
    4.调用proj_process函数,传入经纬度静态文件数据,返回投影后数据
'''
import numpy as np
import h5py
from standard.insert_data_scatter import insert_data2d

import os

class StandardToLatLon(object):
    def __init__(self,lat_0,lat_1,lon_0,lon_1,max_val,resolution=0.04):
        '''
        初始化函数,设置经纬度范围,分辨率
        :param lat_0: 上方纬度,范围为(-90~90)
        :param lat_1: 下方纬度,范围为(-90~90)
        :param lon_0: 左边经度,范围为(-180~180)
        :param lon_1: 右边经度,范围为(-180~180)
        :param resolution:分辨率,度数,默认为0.04
        '''
        self.resolution=float(resolution)
        self.lat_0=float(lat_0)
        self.lat_1=float(lat_1)
        self.lon_0=float(lon_0)
        self.lon_1=float(lon_1)
        self.max_val=max_val


    def coordinate_transform(self, lon, lat):
        '''
        将标称经纬度转为等经纬下经纬度,本函数作为后期兼容性考虑,暂时保留
        :param lon: 经度,需传入numpy数组,范围为[-180,180]
        :param lat: 纬度,需传入numpy数组,范围为[-90，90]
        :return:x,横坐标,对应于经度;y,纵坐标,对应于纬度
        '''
        x=lon
        y=lat
        return x, y
    def proj_process(self,lon,lat,val):
        '''
        投影过程函数
        :param lon:经度数组
        :param lat:纬度数组
        :param val:对应值数组
        :return:投影后数据
        '''
        all_x, all_y = self.coordinate_transform(np.array([self.lon_0, self.lon_1]), np.array([self.lat_0, self.lat_1]))
        if self.lon_0<self.lon_1:
            all_col = (all_x[1] - all_x[0]) / self.resolution
        else:
            all_col = (all_x[1] - all_x[0]+360) / self.resolution
        all_row = (all_y[1] - all_y[0]) / self.resolution
        if self.lon_0<self.lon_1:
            target_index = np.where((lon > self.lon_0) & (lon < self.lon_1) & (lat > self.lat_0) & (lat < self.lat_1))
        else:
            target_index = np.where(((lon > self.lon_0) | (lon < self.lon_1)) & (lat > self.lat_0) & (lat < self.lat_1))
        x, y = self.coordinate_transform(lon[target_index], lat[target_index])

        if self.lon_0<self.lon_1:
            target_col = ((x - all_x[0]) / self.resolution).astype(int)
        else:
            target_col=np.vectorize(lambda x:(x-all_x[0])/self.resolution if x>0 else (x - all_x[0]+360)/self.resolution)(x)
            target_col=target_col.astype(int)
        target_row = ((all_y[1] - y) / self.resolution).astype(int)
        meshg = np.meshgrid(np.arange(int(all_col)), np.arange(int(all_row)))
        grid_z0=insert_data2d(np.hstack((target_row.reshape(-1,1),target_col.reshape(-1,1))),val[target_index],(meshg[1],meshg[0]),)
        #增加填充值及无效值处理
        grid_z0[(np.isnan(grid_z0)) | (grid_z0>self.max_val)] = self.max_val
        return grid_z0

if __name__=='__main__':

    with h5py.File('FY4A_OBI_4000M_NOM_LATLON.HDF',
                   'r') as f:
        s_line=179
        length=1108
        lat = f['Lat'][s_line:s_line+length]
        lon = f['Lon'][s_line:s_line+length]

    hdf = 'Z_SATE_C_BAWX_20200416015623_P_FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200416014918_20200416015335_4000M_V0001.HDF'
    with h5py.File(hdf, 'r') as f:
        cal = f['CALChannel12'][...]
        nom = f['NOMChannel12'][...]

        demo = StandardToLatLon(5, 55, 70, 140, 65535, resolution=0.04)
        res = demo.proj_process(lon, lat, nom)

        with h5py.File(
                '/data2/L1_2/' + os.path.dirname(hdf)[-8:] + '/' + os.path.basename(hdf).split('.')[0] + '_trans.hdf5',
                'w') as f:

            f['CALChannel12'] = cal
            f['NOMChannel12'] = res

