import os
from pathlib import Path

import matplotlib
import metpy
from datetime import datetime
import numpy as np
import pandas as pd  # 数据读取用
import metpy.calc as mpcalc  # 计算一些参数用，比如 抬升凝结高度
from metpy.calc import dry_lapse, moist_lapse, vapor_pressure, dewpoint
from metpy.plots import SkewT  # 画埃玛图
from metpy.units import units  # 单位
import matplotlib.pyplot as plt  # 画图

from service.utils import get_limit_index

matplotlib.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False


# df = pd.read_csv('54511_2019122912.csv')
# # 查看一下数据里面的信息
# df.info()

# file = [r"E:\data\探空\L波段探空\Z_UPAR_I_56691_20220801052526_O_TEMP-L.txt"]
# ele = 'tem'
# title = 'test'
# df = Lliners(file, ele, title).get_pro_data()

# file = ["E:/data/探空/GPS探空/2022083013/EDT_20220830_13.txt"]
# ele = 'tem'
# title = 'test'
# pic = r'D:/tt'
# df = GPSliners(file, ele, title).get_pro_data()


def tlnp(pre, tem, td, hz, ws, wd, up_speed=None, station_id=None, valid_time='YY-MM-DD HH:mm:ss'):
    """
    绘制气象T-lnP热力探空图
    :param pre: 气压
    :param tem: 气温
    :param td: 露点温度
    :param hz: 位势高度
    :param ws: 风速
    :param wd: 风向
    :param up_speed: 上升速度
    :param station_id: 探空站点id
    :param valid_time: 数据观测时间
    :return:
    """
    # 读取各类数据，并进行单位转化
    if up_speed is None:
        up_speed = np.ones_like(pre)
    non_dups = np.array(up_speed) >= 0
    p = np.array(pre)[non_dups] * units.hPa
    T = np.array(tem)[non_dups] * units.degC
    Td = np.array(td)[non_dups] * units.degC
    # Td = mpcalc.dewpoint_from_relative_humidity(T, df['rh'] * units.percent)
    gz = np.array(hz)[non_dups] * units.gpm
    wind_speed = np.array(ws)[non_dups] * units.knots  # 风速
    wind_dir = np.array(wd)[non_dups] * units.degrees  # 方向

    # 得到风的u,v,分量
    u, v = mpcalc.wind_components(wind_speed, wind_dir)
    kindex = mpcalc.k_index(p, T, Td)  # K指数
    try:
        showalter = mpcalc.showalter_index(p, T, Td)  # SI指数
    except ValueError:
        showalter = units.Quantity(np.nan, 'degree_Celsius')

    prof = mpcalc.parcel_profile(p, T[0], Td[0])
    lift_index = mpcalc.lifted_index(p, T, prof)
    try:# Li指数 表示大气层结不稳定，且负值越大，越不稳定
        cape, cin = mpcalc.cape_cin(p, T, Td, prof, which_lfc='top', which_el='most_cape')  # 对流能量值
    except ValueError:
        cape, cin = np.nan, np.nan
    # cape2, cin2 = mpcalc.cape_cin(p, T, Td, prof, which_lfc='top',
    #                          which_el='top')  # 对流能量值
    lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])  # LCL 抬升凝结高度
    ccl_p, ccl_t, t_c = mpcalc.ccl(p, T, Td)  # CCL 对流凝结高度
    lfc_p, _ = mpcalc.lfc(p, T, Td)  # LFC 自由对流高度
    try:
        el_p, _ = mpcalc.el(p, T, Td, prof)  # EL 平衡高度
    except ValueError:
        #  查找数据中非nan值的最小的位置。
        Td_m = np.min([np.max(np.where(np.isfinite(Td))), np.max(np.where(np.isfinite(T)))])
        el_p, _ = mpcalc.el(p[0:Td_m], T[0:Td_m], Td[0:Td_m], prof[0:Td_m])
    zero_temp_index = np.where(T > (0 * units.degC))[0][-1]
    zero_gz = np.mean([gz.m[zero_temp_index], gz.m[zero_temp_index + 1]]) * units.gpm

    fig = plt.figure(figsize=(9, 11))  #
    skew = SkewT(fig, rotation=0)  # rotation=0参数十分重要，代表的意思是温度线与Y轴的夹角 斜温图为45°
    # 画温度层结，露点层结，风
    skew.plot(p, T, 'r', linewidth=1, label='环境温度')
    skew.plot(p, Td, 'g', linewidth=1, label='露点温度')
    inter_y = get_limit_index(p.shape[0], num=50)
    p_lim = np.where(p.m < 100)[0][0]
    skew.plot_barbs(p[0:p_lim][::inter_y], u[0:p_lim][::inter_y], v[0:p_lim][::inter_y],
                    length=6, linewidth=0.8,
                    sizes={"spacing": 0.2, 'height': 0.5, 'width': 0.5, 'emptybarb': 0},
                    barb_increments=dict(half=2, full=4, flag=20))

    # 根据温度和高度画个点，代表LCL
    # skew.plot(lcl_p, lcl_t, 'ko', markerfacecolor='black')
    # 画状态曲线
    prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
    skew.plot(p, prof, 'b', linewidth=1, label='状态曲线')
    # 画能量
    skew.shade_cin(p, T, prof)
    skew.shade_cape(p, T, prof)
    # 画0度线
    # skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
    # 画底图上的干绝热线
    t0 = np.arange(-70, 260, 10) * units.degC
    pressure = units.Quantity(np.arange(1050, 99, -10), 'mbar')
    skew.plot_dry_adiabats(t0=t0, pressure=pressure,
                           alpha=0.25, color='orangered', label='干绝热线', linewidth=0.9)
    # 干绝热线的标记值
    t = dry_lapse(pressure, t0[:, np.newaxis], units.Quantity(1000., 'mbar')).to(units.degC)
    linedata_d = [np.vstack((ti.m, pressure.m)).T for ti in t]
    for i, ld in enumerate(linedata_d):
        if i % 2 != 0:
            continue
        if int(t0[i].m) > 55:
            break
        if i <= 15:
            tex = int(t0[i].m)
            idx = np.where(ld[:, 0] <= -78)[0][0]
            skew.ax.text(ld[idx, 0] * units.degC, ld[idx, 1] * units.hPa, tex,
                         ha='center', va='center', color='orangered', alpha=0.5, fontsize=6, fontweight='bold')
        else:
            tex = int(t0[i].m)
            skew.ax.text(ld[-2, 0] * units.degC, ld[-2, 1] * units.hPa, tex,
                         ha='center', va='center', color='orangered', alpha=0.5, fontsize=6, fontweight='bold')
    # 湿绝热线
    t0 = np.arange(-40, 126, 5) * units.degC
    pressure = units.Quantity(np.arange(1050, 99, -10), 'mbar')
    skew.plot_moist_adiabats(t0=t0, pressure=pressure,
                             alpha=0.25, color='tab:green', label='湿绝热线', linewidth=0.9)
    # 湿绝热线的标记值
    t = moist_lapse(pressure, t0, units.Quantity(1000., 'mbar')).to(units.degC)
    linedata_m = [np.vstack((ti.m, pressure.m)).T for ti in t]
    for i, ld in enumerate(linedata_m):
        if i % 2 != 0:
            continue
        if int(t0[i].m) > 35:
            break
        if i <= 13:
            tex = int(t0[i].m)
            idx = np.where(ld[:, 0] <= -70)[0][0]
            skew.ax.text(ld[idx, 0] * units.degC, ld[idx, 1] * units.hPa, tex,
                         ha='center', va='center', color='blue', alpha=0.5, fontsize=6, fontweight='bold')
        else:
            tex = int(t0[i].m)
            skew.ax.text(ld[-2, 0] * units.degC, ld[-2, 1] * units.hPa, tex,
                         ha='center', va='center', color='blue', alpha=0.5, fontsize=6, fontweight='bold')
    # 饱和比湿线
    mixing_ratio = np.array([0.01, 0.1, 0.4, 1.0, 2.0, 4.0, 7.0, 10, 16, 24, 32, 40, 48, 56])
    pressure = np.arange(1000, 99, -20) * units.hPa
    skew.plot_mixing_lines(mixing_ratio=mixing_ratio / 1000,
                           pressure=np.arange(1000, 99, -20) * units.hPa,
                           linestyle='dotted', color='tab:blue', label='饱和混合比', linewidth=0.7)
    # 饱和比湿线的标记值
    td = dewpoint(vapor_pressure(pressure, (mixing_ratio / 1000).reshape(-1, 1)))
    linedata = [np.vstack((t.m, pressure.m)).T for t in td]
    for i, ld in enumerate(linedata):
        tex = int(mixing_ratio[i]) if mixing_ratio[i] > 1 else mixing_ratio[i]
        skew.ax.text(ld[-2, 0] * units.degC, ld[-2, 1] * units.hPa, tex,
                     ha='center', color='green', fontsize=6, fontweight='bold')

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-80, 40)
    skew.ax.set_ylabel('P/(hPa)')
    skew.ax.set_xlabel('T/(℃)')
    skew.ax.legend(loc='best')
    skew.ax.set_title(f'Station: {station_id} \n', loc='left')  # Position:{df["lon"]}E {df["lat"]}N'
    skew.ax.set_title(f'K:{kindex:.2f}\n SI:{showalter:.2f}\n LI:{lift_index:.2f}\n Cape: {cape:.2f}\n CIN:{cin:.2f}\n'
                      f'LCL_P:{lcl_p.m:.1f} LFC_P:{lfc_p.m:.1f} EL_P:{el_p:.1f}\n ZH:{zero_gz:.1f}\n'
                      f'Valid Time: {valid_time}', loc='right')
    return fig, skew
