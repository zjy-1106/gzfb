# -*- coding: utf-8 -*-
"""
@Time : 2022/7/26 18:54
@Author : YangZhi
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, ticker
from config.config import font_path
import matplotlib as mpl
import matplotlib.dates as mdates

from service.effect_analysis import set_x_time
from service.utils import get_limit_index, ax_add_time_range
from loguru import logger

# from utils.file_uploader import minio_client

mpl.use('Agg')
FONTSIZE = 10


def draw_line_image(x, y, x_label, y_label, title, pic_file):
    xylabel_size = 15
    xticks_size = 0
    figsize = (14, 8)
    unit = "mm"

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['right'].set_visible(False)  # 去掉右边的边框
    ax.spines['top'].set_visible(False)  # 去掉上面的边框

    ax.plot(x, y, linestyle='-', label="label")
    ax.legend(loc='upper right')  # ,bbox_to_anchor=(1,1)

    ax.set_xlabel(x_label, loc='center', size=xylabel_size)
    ax.set_ylabel(y_label, loc='center', size=xylabel_size)
    ax.set_title(title, loc='center', size=xylabel_size)
    plt.xticks(size=xticks_size)
    fig.autofmt_xdate()  # 自动调整角度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=len(x) // 20 if len(x) // 20 != 0 else len(x)))  # 避免横坐标太密
    # plt.show()
    plt.close()


def draw_color_image(x, y, data, unit, x_label, y_label, title, pic_file, levels, colors, x_str, ylims=None,
                     cloud_top=None, cloud_bottom=None, bbbh=None, bbth=None, eff_time=None, hail_time=None):
    """
    绘制填色图
    :param x: 横坐标数据
    :param y: 纵坐标数据
    :param data: 绘图数据
    :param unit: 填色数据的单位
    :param x_label: 横坐标标签
    :param y_label: 纵坐标标签
    :param title: 填色图标题
    :param pic_file: 图片保存地址
    :param levels: 填色等级划分列表
    :param colors: 填色色值列表
    :param cloud_top: 云顶高度数据
    :param cloud_bottom: 云底高度数据
    :return: None
    """
    # if os.path.isfile(pic_file):
    #     return pic_file

    file_path = os.path.dirname(pic_file)
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    data[data <= -999] = np.nan
    # font_manager.fontManager.addfont(font_path)
    # prop = font_manager.FontProperties(fname=font_path)
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = prop.get_name()  # 使图正常显示中文
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rc("font", family='DengXian')
    plt.rcParams['axes.unicode_minus'] = False  # 使刻度正常显示正负号

    fig, ax = plt.subplots(figsize=(8, 3))
    if title:
        ax.set_title(title, fontsize=FONTSIZE)
    ax.set_ylabel(y_label, loc='center', fontsize=FONTSIZE)
    # ax.set_xlabel(x_label)

    # 自定义自适应x轴（指定x轴显示固定个数的坐标点，这里固定最多显示10个横坐标点）
    # step = get_limit_index(len(x))
    # plt.xticks(x[::step], x_str[::step])
    tm = None if int((x[-1] - x[0]).seconds / 60) < 8 else int((x[-1] - x[0]).seconds / 60)
    kwargs_major, kwargs_minor, timestyle = set_x_time(x, tm=tm)
    majorformatter = mdates.DateFormatter(timestyle)
    rule1 = mdates.rrulewrapper(**kwargs_major)
    loc1 = mdates.RRuleLocator(rule1)
    ax.xaxis.set_major_locator(loc1)
    ax.xaxis.set_major_formatter(majorformatter)
    rule = mdates.rrulewrapper(**kwargs_minor)
    loc = mdates.RRuleLocator(rule)
    ax.xaxis.set_minor_locator(loc)

    fig.autofmt_xdate(rotation=45)  # 自动调整x轴标签的角度

    Y, X = np.meshgrid(y, x)
    # cset = ax.contourf(X, Y, data, levels, colors=colors)
    cset = ax.pcolormesh(X, Y, data, vmin=levels[0], vmax=levels[-1], cmap=colors, shading='auto')
    cb = plt.colorbar(cset, fraction=0.027, pad=0.02, aspect=30)
    cb.ax.set_title(unit, fontsize=8)
    cb.ax.tick_params(labelsize=8)
    ax.set_ylim(ylims)
    plt.tick_params(labelsize=FONTSIZE)
    ax_add_time_range(ax, eff_time, alpha=0.6, color='w')
    ax_add_time_range(ax, hail_time, alpha=0.4, color='k')

    if len(cloud_bottom) > 0:
        print("===" * 8)
        ax1 = ax.plot(x, cloud_bottom, 'k^', markersize=2, label='云底')
        ax2 = ax.plot(x, cloud_top, 'm^', markersize=2, label='云顶')
        ax3 = ax.plot(x, bbbh, color='black', linewidth=0.8, label='零度层底')
        ax4 = ax.plot(x, bbth, color='purple', linewidth=0.8, label='零度层顶')
        ax.set_title(f'{title}', fontsize=FONTSIZE)
        ax.legend(handles=[ax1[0], ax2[0], ax3[0], ax4[0]],
                  labels=['云底', '云顶', '零度层底', '零度层顶'],
                  loc=1, ncol=2, frameon=False)

    plt.savefig(pic_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    # 上传图片
    # # minio_client.put_object(pic_file)

# if __name__ == '__main__':
#     # colors = ('#000000', '#0000bb', '#04999a', '#00aaaa', '#00aaa9', '#02b7b6', '#00c2c0', '#03cbca', '#00d4d4',
#     #           '#008800', '#00bb00', '#00d800', '#999900', '#a9a900', '#b7b700', '#c2c107', '#cccb03', '#d4d400',
#     #           '#dcdc00', '#bb0000', '#d80000', '#980098', '#a800a8', '#b500b5', '#bf00bf', '#c900c9', '#d100d1',
#     #           '#d800d8')
#     # levels = (
#     #     10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000,
#     #     5000, 6000, 7000, 8000, 9000, 10000)
#     colors = ('#000000', '#0000a6', '#0000c7', '#009393', '#00b3b3', '#00d2d2', '#009d00', '#00bd00', '#888800',
#               '#a7a700', '#c7c700', '#920000', '#b20000', '#d20000', '#9c009c', '#ba00ba', '#d800d8')
#     levels = (-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40)
#     data_files = ["/Users/yangzhi/code/GZFB/code/radar_analysis_pic/out/CWR/20220520000054CWR_prod.dat",
#                   "/Users/yangzhi/code/GZFB/code/radar_analysis_pic/out/CWR/20220520000057CWR_prod.dat",
#                   "/Users/yangzhi/code/GZFB/code/radar_analysis_pic/out/CWR/20220520000101CWR_prod.dat",
#                   "/Users/yangzhi/code/GZFB/code/radar_analysis_pic/out/CWR/20220520000104CWR_prod.dat"]
#     cwr = CWRData(data_files)
#     raw_ref = cwr.raw_ref
#     title = "test"
#     pic_file = "./aaa.png"
#     data = np.array(raw_ref, dtype=float)
#     data[data == -999] = np.nan
#     x = cwr.time_list
#     x = [(t - x[0]).total_seconds() for t in x]
#     y = list(range(1, data.shape[1] + 1))
#     x_label = "x_label"
#     y_label = "y_label"
#     unit = "dBZ"
#     # print(np.unique(data))
#     draw_color_image(x, y, data, unit, x_label, y_label, title, pic_file, levels, colors)
#     # draw_line_image(x, data[:, 2], x_label, y_label, title, pic_file)
