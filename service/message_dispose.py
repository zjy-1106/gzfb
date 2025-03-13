#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消息处理类
"""
import time

from loguru import logger
from abc import ABCMeta, abstractmethod

from mq.pubulisher import publish_complete_msg
from service.FY4A import FY4A_analysis
from service.cr_cwr import CrCwr
from service.dda_wind import DDAWindController
from service.dsd_rainrate_fusion import DsdRainRateController
from service.hail_area import Hail
from service.mrr import MrrController
from service.dsd import DsdController

from service.mwr import MWRController
from service.qc_analysis import QCReportDraw
from service.radar_section import Section
from service.cwr import CWRController
from service.cr import CRController
from service.ref import Reflectivity
from service.l_liner import Lliners
from service.gps_liner import GPSliners
from service.standard_rad import Standard_rad
from service.wind import Wind_profile
from service.effect_analysis import Effect_Analysis
from service.spatial import Spatial_3D


class Dispose(metaclass=ABCMeta):
    @abstractmethod
    def handle_request(self, request):
        pass


class MRR(Dispose):
    def handle_request(self, request):
        # 微雨雷达
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        height = message["arithmeticParams"]["equipParam"].get('height')
        element = message["arithmeticParams"]["equipParam"]['element']
        start_t = message["arithmeticParams"]["equipParam"]['startTime']
        end_t = message["arithmeticParams"]["equipParam"]['endTime']
        title = message["arithmeticParams"]['title']
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        try:
            ylims = message["arithmeticParams"]['height_limit']
        except Exception:
            ylims = None
        obj = MrrController(input_filelist, height, jobid, element, start_t, end_t)
        cell_png = obj.run(title, ylims, eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class DSD(Dispose):
    def handle_request(self, request):
        # 雨滴谱
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        element = message["arithmeticParams"]["equipParam"].get('element')
        start_t = message["arithmeticParams"]["equipParam"]['startTime']
        end_t = message["arithmeticParams"]["equipParam"]['endTime']
        density_time = message["arithmeticParams"]["equipParam"].get('dataTime')
        title = message["arithmeticParams"]['title']
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        obj = DsdController(input_filelist, jobid, element, start_t, end_t, density_time)
        cell_png = obj.run(title, eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class RadarSection(Dispose):
    def handle_request(self, request):
        # 雷达剖面
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        element = message["arithmeticParams"]["equipParam"].get('element')
        elements = message["arithmeticParams"]["equipParam"].get('elements')
        img_type = message["arithmeticParams"].get('img_type')  # 图片类型
        title = message["arithmeticParams"].get('title')  # 图片标题
        start_lon, start_lat = message["arithmeticParams"].get('start_point', ('', ''))  # 起始点坐标
        end_lon, end_lat = message["arithmeticParams"].get('end_point', ('', ''))  # 结束点坐标
        center_point = message["arithmeticParams"].get('mid_point')  # 冰雹质心位置
        elevation = message["arithmeticParams"]["equipParam"].get('elevation')  # 仰角
        distance = message["arithmeticParams"]["equipParam"].get('distance')  # 距离库数
        angle = message["arithmeticParams"]["equipParam"].get('angle')  # 径向剖面相位角
        blast_point_list = message["arithmeticParams"].get('blast_point_list')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        # blast_lon, blast_lat = message["arithmeticParams"].get('blast_start_point', ('', ''))
        # blast_end = message["arithmeticParams"].get('blast_end_point')
        # blast_h = message["arithmeticParams"].get('blast_height')
        # 参数：
        obj = Section(input_filelist, element, img_type, jobid, title, elements=elements)
        cell_png = obj.run(start_lat, end_lat, start_lon, end_lon, angle, blast_point_list, elevation, distance,
                           center_point, eff_time=eff_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))


class DsdRainRate(Dispose):
    def handle_request(self, request):
        # 雨滴谱反演，降水估测
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        element = message['arithmeticParams'].get('element')
        point_lat = message['arithmeticParams'].get('lat')
        point_lon = message['arithmeticParams'].get('lon')
        elevation = message['arithmeticParams']['equipParam'].get('height')
        threshold = message['arithmeticParams']['equipParam'].get('threshold')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        obj = DsdRainRateController(input_filelist, element, elevation, jobid, threshold)
        cell_png = obj.run(point_lat, point_lon, eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class DDAWind(Dispose):
    def handle_request(self, request):
        # 三维风场
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        height = message['arithmeticParams']['equipParam'].get('height')
        img_type = message["arithmeticParams"]['equipParam'].get('img_type')  # 图片类型
        title = message["arithmeticParams"]['equipParam'].get('title')  # 图片类型
        point_lat = message['arithmeticParams']['equipParam'].get('lat')  # 单点分析参数
        point_lon = message['arithmeticParams']['equipParam'].get('lon')
        is_part = message['arithmeticParams']['equipParam'].get('is_part')  # 是否画区域 0，1
        mid_point = message['arithmeticParams']['equipParam'].get('mid_point')  # 中心剖面
        start_point = message["arithmeticParams"]['equipParam'].get('start_point')  # 起始点坐标
        end_point = message["arithmeticParams"]['equipParam'].get('end_point')  # 结束点坐标
        work_point = message["arithmeticParams"]['equipParam'].get('work_point')  # 炮点点坐标
        blast_point = message["arithmeticParams"]['equipParam'].get('blast_point')  # 爆炸点坐标
        blast_height = message['arithmeticParams']['equipParam'].get('blast_height')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        threshold = message['arithmeticParams']['equipParam'].get('threshold')  # 阈值
        elements = message["arithmeticParams"]["equipParam"].get('elements')
        obj = DDAWindController(input_filelist, height, jobid, title, threshold)
        cell_png = obj.run(img_type, point_lon, point_lat, start_point, end_point, mid_point, is_part,
                           blast_point, work_point, eff_time, blast_height, hail_time=hail_time, eles=elements)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class CWR(Dispose):
    def handle_request(self, request):
        # c波段连续波雷达
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        height = message['arithmeticParams']['equipParam'].get('height')
        element = message['arithmeticParams']['equipParam'].get('element')
        data_time = message['arithmeticParams']['equipParam'].get('dataTime')
        title = message['arithmeticParams'].get('title')
        analysis_type = message['arithmeticParams']['equipParam'].get('analysis_type')
        start_time = message['arithmeticParams']['equipParam'].get('startTime')
        end_time = message['arithmeticParams']['equipParam'].get('endTime')
        is_picture = message['arithmeticParams']['equipParam'].get('picture')
        ylims = message['arithmeticParams']['equipParam'].get('height_limit')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        if not ylims:
            ylims = None
        obj = CWRController(input_filelist, height, jobid, element, title, analysis_type, ylims, is_picture, data_time,
                            start_time, end_time)
        cell_png = obj.run(eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class MWR(Dispose):
    def handle_request(self, request):
        # 微波辐射计
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        height = message["arithmeticParams"]["equipParam"].get('height')
        start_t = message["arithmeticParams"]["equipParam"].get('startTime')
        end_t = message["arithmeticParams"]["equipParam"].get('endTime')
        data_t = message["arithmeticParams"]["equipParam"].get('dataTime')
        ele = message["arithmeticParams"]["equipParam"].get('element')
        eles = message["arithmeticParams"]["equipParam"].get('elements')
        title = message['arithmeticParams']['equipParam'].get('title')
        analysis_type = message['arithmeticParams']['equipParam'].get('analysis_type')
        ylims = message['arithmeticParams']['equipParam'].get('height_limit')
        if not ylims:
            ylims = None
        station_and_equip_name = message['arithmeticParams']['equipParam'].get('station_and_equip_name')
        product_names = message['arithmeticParams']['equipParam'].get('product_names')
        iscorrect = message['arithmeticParams']['equipParam'].get('is_correct')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        obj = MWRController(input_filelist, jobid, start_t, end_t, ele, height, data_t, eles)
        cell_png = obj.run(title, analysis_type, ylims, station_and_equip_name, product_names, iscorrect,
                           eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class CR(Dispose):
    def handle_request(self, request):
        # 云雷达
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        height = message['arithmeticParams']['equipParam'].get('height')
        elevation = message['arithmeticParams']['equipParam'].get('elevation')
        point_lat = message['arithmeticParams']['equipParam'].get('lat')
        point_lon = message['arithmeticParams']['equipParam'].get('lon')
        element = message['arithmeticParams']['equipParam'].get('element')
        title = message['arithmeticParams']['equipParam'].get('title')
        scan = message['arithmeticParams']['equipParam'].get('scan_model')
        analysis_type = message['arithmeticParams']['equipParam'].get('analysis_type')
        ylims = message['arithmeticParams']['equipParam'].get('height_limit')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        if not ylims:
            ylims = None
        obj = CRController(input_filelist, jobid, element, title, scan, height, elevation, point_lat, point_lon,
                           analysis_type, ylims)
        cell_png = obj.run(eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class REF(Dispose):
    def handle_request(self, request):
        # 基本反射率区域分析
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        element = message['arithmeticParams']['equipParam'].get('element')  # 要素
        elements = message['arithmeticParams']['equipParam'].get('elements')  # 要素
        point_lat = message['arithmeticParams']['equipParam'].get('lat')  # 点纬度
        point_lon = message['arithmeticParams']['equipParam'].get('lon')  # 点经度
        str_point = message['arithmeticParams']['equipParam'].get('start_point')  # 起始点坐标
        end_point = message['arithmeticParams']['equipParam'].get('end_point')  # 结束点坐标
        mid_point = message['arithmeticParams']['equipParam'].get('mid_point')  # 冰雹中心点坐标
        title = message['arithmeticParams']['equipParam'].get('title')
        analysis_type = message['arithmeticParams']['equipParam'].get('analysis_type')  # 绘图类型 1，6
        height = message['arithmeticParams']['equipParam'].get('height')  # 绘图高度层 0-149
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        obj = Reflectivity(input_filelist, jobid, element, title, analysis_type, height, elements=elements)
        cell_png = obj.run(point_lat, point_lon, str_point, end_point, mid_point, eff_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))


class Lliner(Dispose):
    def handle_request(self, request):
        # L波段探空
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        element = message['arithmeticParams']['equipParam'].get('element')  # 要素
        title = message['arithmeticParams']['equipParam'].get('title')
        obj = Lliners(input_filelist, element, title)
        cell_png = obj.run()
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class GPSliner(Dispose):
    def handle_request(self, request):
        # GPS波段探空
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        element = message['arithmeticParams']['equipParam'].get('element')  # 要素
        title = message['arithmeticParams']['equipParam'].get('title')
        obj = GPSliners(input_filelist, element, title)
        cell_png = obj.run()
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class RDwind(Dispose):
    def handle_request(self, request):
        # 风廓线雷达
        work_start = time.time()
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        img_type = message["arithmeticParams"]['equipParam'].get('img_type')  # 图片类型
        title = message['arithmeticParams']['equipParam'].get('title')
        ylims = message['arithmeticParams']['equipParam'].get('height_limit')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        if not ylims:
            ylims = None
        obj = Wind_profile(input_filelist, jobid, img_type)
        cell_png = obj.run(title, ylims, eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class EffectAnalysis(Dispose):
    def handle_request(self, request):
        # 效果分析
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        height = message['arithmeticParams']['equipParam'].get('heights')  # 极坐标高度范围
        azimuth = message['arithmeticParams']['equipParam'].get('azimuth_range')  # 极坐标方位角范围
        start_point = message['arithmeticParams']['equipParam'].get('start_point')  # 网格化区域起始点
        end_point = message['arithmeticParams']['equipParam'].get('end_point')  # 网格化区域结束点
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        element = message['arithmeticParams']['equipParam'].get('element')  # 统计要素
        # statistic = message['arithmeticParams']['equipParam'].get('statistic_code')  # 统计类目【max, mean, area】
        title = message['arithmeticParams']['equipParam'].get('title')
        analysis_type = message['arithmeticParams']['equipParam'].get('analysis_type')  # 统计类别
        sequence_type = message['arithmeticParams']['equipParam'].get('sequence_type')
        statistic_args = message['arithmeticParams']['equipParam'].get('statistic_args')
        obj = Effect_Analysis(input_filelist, jobid, element, title, analysis_type)
        cell_png = obj.run(start_point, end_point, eff_time, hail_time, height, azimuth, sequence_type, statistic_args)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))


class WeatherRadar(Dispose):
    def handle_request(self, request):
        # 天擎雷达数据解析
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        start_point = message['arithmeticParams']['equipParam'].get('start_point')  # 网格化区域起始点
        end_point = message['arithmeticParams']['equipParam'].get('end_point')  # 网格化区域结束点
        element = message['arithmeticParams']['equipParam'].get('element')  # 统计要素
        elevation = message['arithmeticParams']['equipParam'].get('elevation')  # 仰角
        azimuth = message['arithmeticParams']['equipParam'].get('angle')  # 方位角
        title = message['arithmeticParams']['equipParam'].get('title')
        scan = message['arithmeticParams']['equipParam'].get('analysis_type')  # 扫描模式
        threshold = message['arithmeticParams']['equipParam'].get('threshold')  # 阈值
        obj = Standard_rad(input_filelist, element, jobid, threshold, elevation, azimuth)
        cell_png = obj.run(scan, start_point, end_point, title)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))


class FY4A(Dispose):
    def handle_request(self, request):
        # FY4A卫星云图
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        classes = message['arithmeticParams']['equipParam'].get('classes')
        obj = FY4A_analysis(input_filelist, jobid, classes)
        cell_png = obj.run()
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))


class Spaticl(Dispose):
    def handle_request(self, request):
        # 空间三维展示图
        message = request.requestContent
        input_filelist = message['jobFileDtoList']
        jobid = message['jobId']
        groupCode = message['arithmeticParams'].get('groupCode')
        start_point = message['arithmeticParams']['equipParam'].get('start_point')  # 网格化区域起始点
        end_point = message['arithmeticParams']['equipParam'].get('end_point')  # 网格化区域结束点
        level = message['arithmeticParams']['equipParam'].get('height')
        element = message['arithmeticParams']['equipParam'].get('element')
        obj = Spatial_3D(input_filelist, element, jobid, level)
        cell_png = obj.run(ele_type=groupCode, start_point=start_point, end_point=end_point)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))


class FusionCloudRain(Dispose):
    def handle_request(self, request):
        # 云降水融合
        work_start = time.time()
        message = request.requestContent
        jobid = message['jobId']
        start_t = message["arithmeticParams"]["equipParam"].get('startTime')
        end_t = message["arithmeticParams"]["equipParam"].get('endTime')
        eff_time = message['arithmeticParams']['equipParam'].get('time_range')  # 作业时间点
        hail_time = message['arithmeticParams']['equipParam'].get('damage_time_range')  # 灾损时间点
        obj = CrCwr(jobid, start_t, end_t)
        cell_png = obj.run(eff_time=eff_time, hail_time=hail_time)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class HailArea(Dispose):
    def handle_request(self, request):
        # 云降水融合
        work_start = time.time()
        message = request.requestContent
        jobid = message['jobId']
        input_filelist = message['jobFileDtoList']
        point_dict = message["arithmeticParams"]["equipParam"].get('point_list')
        obj = Hail(input_filelist, jobid)
        cell_png = obj.run(point_dict)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class QCReport(Dispose):
    def handle_request(self, request):
        # 质控报告
        work_start = time.time()
        message = request.requestContent
        jobid = message['jobId']
        input_filelist = message['jobFileDtoList']
        # ele = message['arithmeticParams']['equipParam'].get('ele')  # 要素
        # img_type = message['arithmeticParams']['equipParam'].get('img_type')  # 绘图类型
        azimuth = message['arithmeticParams']['equipParam'].get('elevations')  # 仰角
        angle = message['arithmeticParams']['equipParam'].get('angle')  # 方位角
        obj = QCReportDraw(input_filelist, jobid)
        cell_png = obj.run(azimuth, angle)
        publish_complete_msg("20", jobid, request.image_name, cell_png)
        logger.info("{}算法执行完成，产品图已生成".format(request.image_name))
        print(f'========运行耗时{time.time() - work_start} S========')


class Message:
    image_name = ''
    requestContent = None
