#!/usr/bin/env python
# -*- coding: utf-8 -*-
import yaml
import os

font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "font/SourceHanSansSC-Bold.otf")

ELEMENT_MAP_DSD = {
    "raininess": {
        "title": "时间-雨强",
        "unit": "mm/h",
    },
    "rainstrong": {
        "title": "时间-雨强",
        "unit": "mm/h",
    },
    "water": {
        "title": "时间-含水量",
        "unit": "g/m³",
    },
    "echo": {
        "title": "时间-回波强度ZH",
        "unit": "dBZ",
    },
    "echobefoce": {
        "title": "时间-回波强度ZH",
        "unit": "dBZ",
    },
    "ZDR": {
        "title": "时间-ZDR",
        "unit": "dB",
    },
    "KDP": {
        "title": "时间-KDP",
        "unit": "°/km",
    },
    "Lattenuation": {
        "title": "时间-水平偏振衰减系数",
        "unit": "dB/km",
    },
    "Vattenuation": {
        "title": "时间-垂直偏振波的衰减系数",
        "unit": "dB/km",
    },
}

ELEMENT_MAP_MRR = {
    "raininess": {
        "title": "时间-雨强",
        "unit": "mm/h",
    },
    "water": {
        "title": "时间-含水量",
        "unit": "g/m³",
    },
    "echo": {
        "title": "时间-回波强度Z",
        "unit": "dBZ",
    },
    "echos": {
        "title": "时间-回波强度(Ka、Ku、X、C、S)",
        "unit": "dBZ",
    },
    "ZDR": {
        "title": "时间-反射率因子(ZDR)",
        "unit": "dB",
    },
    "density": {
        "title": "时间-雨滴谱分布参数(数密度)",
        "unit": "mm-¹m-³",
    },
    "diameter": {
        "title": "时间-雨滴谱分布参数(平均直径)",
        "unit": "mm",
    },
    "Mu": {
        "title": "时间-雨滴谱分布参数(形状因子)",
        "unit": " ",
    },
}

# c波段连续波雷达产品图配置信息
c_pic_config = {
    "raw_ref": {"title": "原始回波强度时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "dBZ",
                "vlim": [-40, 40],
                "cmap": 'jet',
                "name": "回波强度"
                },
    "raw_vel": {"title": "原始径向速度时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "m/s",
                "vlim": [-20, 20],
                "cmap": 'jet',
                "name": "径向速度"
                },
    "raw_sw": {"title": "原始速度谱宽时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "m/s",
               "vlim": [0, 15],
               "cmap": 'jet',
               "name": "速度谱宽"
               },
    "qc_ref": {"title": "质控后回波强度时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "dBZ",
               "vlim": [-40, 40],
               "cmap": 'jet',
               "name": "回波强度"
               },
    "qc_vel": {"title": "质控后径向速度时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "m/s",
               "vlim": [-20, 20],
               "cmap": 'jet',
               "name": "径向速度"
               },
    "qc_sw": {"title": "质控后速度谱宽时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "m/s",
              "vlim": [0, 15],
              "cmap": 'jet',
              "name": "速度谱宽"
              },
    "vair": {"title": "空气上升速度时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "m/s",
             "vlim": [-25, 25],
             "cmap": 'jet',
             "name": "空气上升速度"
             }
}
c_line_config = {
    "raw_ref": {"element": "ref", "yname": ["原始回波强度时序图"], "xlabel": "时间", "ylabel": "dBZ", "picType": 0},
    "raw_vel": {"element": "vel", "yname": ["原始径向速度时序图"], "xlabel": "时间", "ylabel": "m/s", "picType": 0},
    "raw_sw": {"element": "sw", "yname": ["原始速度谱宽时序图"], "xlabel": "时间", "ylabel": "m/s", "picType": 0},
    "qc_ref": {"element": "ref", "yname": ["质控后回波强度时序图"], "xlabel": "时间", "ylabel": "dBZ", "picType": 1},
    "qc_vel": {"element": "vel", "yname": ["质控后径向速度时序图"], "xlabel": "时间", "ylabel": "m/s", "picType": 1},
    "qc_sw": {"element": "sw", "yname": ["质控后速度谱宽时序图"], "xlabel": "时间", "ylabel": "m/s", "picType": 1},
    "vair": {"element": "vair", "yname": ["空气上升速度时序图"], "xlabel": "时间", "ylabel": "m/s", "picType": 1}
}
c_profile_config = {
    "raw_ref": {"element": "ref", "yname": ["原始回波强度廓线图"], "xlabel": "(dBZ)", "ylabel": "高度(km)", "picType": 0},
    "qc_ref": {"element": "ref", "yname": ["质控后回波强度廓线图"], "xlabel": "(dBZ)", "ylabel": "高度(km)", "picType": 1},
    "raw_vel": {"element": "vel", "yname": ["原始径向速度廓线图"], "xlabel": "(m/s)", "ylabel": "高度(km)", "picType": 0},
    "qc_vel": {"element": "vel", "yname": ["质控后径向速度廓线图"], "xlabel": "(m/s)", "ylabel": "高度(km)", "picType": 1},
    "raw_sw": {"element": "sw", "yname": ["原始速度谱宽廓线图"], "xlabel": "(m/s)", "ylabel": "高度(km)", "picType": 0},
    "qc_sw": {"element": "sw", "yname": ["质控后速度谱宽廓线图"], "xlabel": "(m/s)", "ylabel": "高度(km)", "picType": 1},
    "vair": {"element": "vair", "yname": ["空气上升速度廓线图"], "xlabel": "(m/s)", "ylabel": "高度(km)", "picType": 1}
}
# 色标
colors = ('#000000', '#0000bb', '#04999a', '#00aaaa', '#00aaa9', '#02b7b6', '#00c2c0', '#03cbca', '#00d4d4',
          '#008800', '#00bb00', '#00d800', '#999900', '#a9a900', '#b7b700', '#c2c107', '#cccb03', '#d4d400',
          '#dcdc00', '#bb0000', '#d80000', '#980098', '#a800a8', '#b500b5', '#bf00bf', '#c900c9', '#d100d1',
          '#d800d8')
levels = (
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000,
    5000, 6000, 7000, 8000, 9000, 10000)

cr_pic_config = {
    "ref": {"title": "回波强度时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "dBZ",
            "colors": (
                '#000080', '#0000c8', '#0000ff', '#0040ff', '#0080ff', '#00bfff', '#15ffe2', '#48ffaf', '#7bff7b',
                '#afff48', '#e2ff15', '#ffd200', '#ff9700', '#ff5c00', '#ff2100', '#c80000', '#800000'),
            "levels": (-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40)
            },
    "vel": {"title": "径向速度时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "m/s",
            "colors": (
                '#0000c8', '#0000ff', '#0040ff', '#0080ff', '#00bfff', '#15ffe2', '#48ffaf', '#afff48', '#e2ff15',
                '#ffd200', '#ff9700', '#ff5c00', '#ff2100', '#c80000'),
            "levels": (-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12)
            },
    "sw": {"title": "速度谱宽时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "m/s",
           "colors": (
               '#000080', '#0000f3', '#004dff', '#00b3ff', '#29ffce', '#7bff7b', '#ceff29', '#ffc600', '#ff6800',
               '#f30900', '#800000'),
           "levels": (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)  # 11, 12, 13, 14, 15, 16
           },
    "IWC": {"title": "冰云含量时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "g/m${^3}$",
            "colors": (
                '#1668d5', '#05bcfc', '#7bf8fe', '#0a9139', '#0ad509', '#ade29e', '#e5b93b', '#f5cf43', '#ffdf68',
                '#fa0408', '#fa6061', '#fbc0cc'
            ),
            "levels": (0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 2)
            },
    "LWC": {"title": "液体含水量时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "g/m${^3}$",
            "colors": (
                '#1668d5', '#05bcfc', '#7bf8fe', '#0a9139', '#0ad509', '#ade29e', '#e5b93b', '#f5cf43', '#ffdf68',
                '#fa0408', '#fa6061', '#fbc0cc'
            ),
            "levels": (0.001, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15)
            },
    "RWC": {"title": "雨含水量时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "g/m${^3}$",
            "colors": (
                '#1668d5', '#05bcfc', '#7bf8fe', '#0a9139', '#0ad509', '#ade29e', '#e5b93b', '#f5cf43', '#ffdf68',
                '#fa0408', '#fa6061', '#fbc0cc'
            ),
            "levels": (0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 2)
            },
    "SWC": {"title": "雪含水量时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "g/m${^3}$",
            "colors": (
                '#1668d5', '#05bcfc', '#7bf8fe', '#0a9139', '#0ad509', '#ade29e', '#e5b93b', '#f5cf43', '#ffdf68',
                '#fa0408', '#fa6061', '#fbc0cc'
            ),
            "levels": (0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 2)
            },
}
cr_line_config = {
    "ref": {"element": "ref", "yname": ["回波强度垂直廓线"], "xlabel": "dBZ", "ylabel": "高度(km)", "picType": 1},
    "vel": {"element": "vel", "yname": ["径向速度垂直廓线"], "xlabel": "m/s", "ylabel": "高度(km)", "picType": 1},
    "sw": {"element": "sw", "yname": ["速度谱宽垂直廓线"], "xlabel": "m/s", "ylabel": "高度(km)", "picType": 1},
    "IWC": {"element": "IWC", "yname": ["冰云含量垂直廓线"], "xlabel": "g/m³", "ylabel": "高度(km)", "picType": 1},
    "LWC": {"element": "LWC", "yname": ["液体含水量垂直廓线"], "xlabel": "g/m³", "ylabel": "高度(km)", "picType": 1},
    "RWC": {"element": "RWC", "yname": ["雨含水量垂直廓线"], "xlabel": "g/m³", "ylabel": "高度(km)", "picType": 1},
    "SWC": {"element": "SWC", "yname": ["雪含水量垂直廓线"], "xlabel": "g/m³", "ylabel": "高度(km)", "picType": 1}
}

cr_cwr_pic_config = {
    "CR_Ref": {"title": "回波强度时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "dBZ",
               "colors": (
                   '#000080', '#0000c8', '#0000ff', '#0040ff', '#0080ff', '#00bfff', '#15ffe2', '#48ffaf', '#7bff7b',
                   '#afff48', '#e2ff15', '#ffd200', '#ff9700', '#ff5c00', '#ff2100', '#c80000', '#800000'),
               "levels": (-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40)
               },
    "CWR_Ref": {"title": "质控后回波强度时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "dBZ",
                "vlim": [-40, 40],
                "cmap": 'jet'
                },
    "LWC": {"title": "液体含水量时间高度图", "x_label": "时间", "y_label": "高度(km)", "unit": "g/m${^3}$",
            "colors": (
                '#1668d5', '#05bcfc', '#7bf8fe', '#0a9139', '#0ad509', '#ade29e', '#e5b93b', '#f5cf43', '#ffdf68',
                '#fa0408', '#fa6061', '#fbc0cc'
            ),
            "levels": (0.001, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15)
            },
    "vair": {"title": "空气上升速度时间-高度图", "x_label": "time", "y_label": "高度(km)", "unit": "m/s",
             "vlim": [-25, 25],
             "cmap": 'jet'
             }
}


class Config(object):
    def __init__(self):
        self._config = None
        # 判断运行平台，是否拉取配置文件
        # 加载本地配置文件
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    @property
    def get_mrr_config(self):
        return self._config.get('mrr')

    @property
    def get_dsd_config(self):
        return self._config.get('dsd')

    @property
    def get_mwr_config(self):
        return self._config.get('mwr')

    @property
    def get_path(self):
        return self._config.get('path')

    @property
    def get_cwr_config(self):
        return self._config.get('cwr')

    @property
    def get_cr_config(self):
        return self._config.get('cr')

    @property
    def get_dsd_rainrate_config(self):  # 雨滴谱反演配置
        return self._config.get('estimate_rainrate')

    @property
    def get_radar_section_config(self):
        return self._config.get('radar_section')

    @property
    def get_dda_wind_config(self):
        return self._config.get('dda_wind')

    @property
    def get_rabbitmq_cfg(self):
        return self._config.get('rabbitmq')

    @property
    def get_ref_cfg(self):
        return self._config.get('ref')

    @property
    def get_wind_cfg(self):
        return self._config.get('wind')

    @property
    def get_eff_cfg(self):
        return self._config.get('effect_analysis')

    @property
    def get_minio_cfg(self):
        # 获取minio文件上传配置
        return self._config.get('minio')

    @property
    def get_std_rad_cfg(self):
        return self._config.get('standard')

    @property
    def get_FY4A_cfg(self):
        return self._config.get('fy4a')

    @property
    def get_spatial_cfg(self):
        return self._config.get('spatial')

    @property
    def get_ltk_cfg(self):
        return self._config.get('ltk')

    @property
    def get_gpstk_cfg(self):
        return self._config.get('gpstk')

    @property
    def get_crcwr_cfg(self):
        return self._config.get('cr_cwr')

    @property
    def get_hail_cfg(self):
        return self._config.get('hail')

    @property
    def get_qc_cfg(self):
        return self._config.get('report')

    @property
    def get_data_server_cfg(self):
        return self._config.get('server')


config_info = Config()
