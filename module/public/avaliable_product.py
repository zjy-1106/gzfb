#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   支持的产品类型
"""

dsd_image_type_ori = {
    "diameter_speed_density": ("质控后雨滴谱随雨滴直径和下落速度变化图", "mm${^-}$${^1}$·m${^-}$${^3}$"),
    "time_raindrops_density": ("质控前雨滴谱随时间和雨滴大小变化图", "mm${^-}$${^1}$·m${^-}$${^3}$"),
    "time_rainspeed_density": ("质控前雨滴谱随时间和速度变化图", "mm${^-}$${^1}$·m${^-}$${^3}$"),
}

dsd_image_type = {
    "diameter_speed_density": ("质控后雨滴谱随雨滴直径和下落速度变化图", "mm${^-}$${^1}$·m${^-}$${^3}$"),
    "time_raindrops_density": ("质控后雨滴谱随时间和雨滴大小变化图", "mm${^-}$${^1}$·m${^-}$${^3}$"),
    "time_rainspeed_density": ("质控后雨滴谱随时间和速度变化图", "mm${^-}$${^1}$·m${^-}$${^3}$")
}

mrr_image_type = {
    'time_height_echo': ("时间-高度-衰减订正后的回波强度", "dBZ"),
    'time_height_Ka': ("时间-高度-Ka回波强度", "dBZ"),
    'time_height_Ku': ("时间-高度-Ku回波强度", "dBZ"),
    'time_height_X': ("时间-高度-X回波强度", "dBZ"),
    'time_height_C': ("时间-高度-C回波强度", "dBZ"),
    'time_height_S': ("时间-高度-S回波强度", "dBZ"),
    'time_height_raininess': ('时间高度-雨强', 'mm/h'),
    'time_height_water': ('时间高度-含水量', 'g/m${^3}$'),
    'time_height_density': ('时间高度-Gamma分布的数密度Nw', 'mm${^-}$${^1}$·m${^-}$${^3}$'),
    'time_height_diameter': ('时间高度-Gamma分布的平均直径Dm', "mm"),
    'time_height_Mu': ('时间高度-Gamma分布的形状因子Mu', ''),
    'time_diameter_density': ("时间-雨滴数大小-雨滴数密度变化图", "mm${^-}$${^1}$·m${^-}$${^3}$"),
    'diameter_height_density': ("雨滴大小-高度-雨滴数密度变化图", "mm${^-}$${^1}$·m${^-}$${^3}$")}

mrr_image_type_ori = {
    'time_height_echo': ("时间-高度-回波强度", "dBZ"),
    'time_height_raininess': ('时间高度-雨强', 'mm/h'),
    'time_height_water': ('时间高度-含水量', 'g/m${^3}$'),
    'time_diameter_density': ("时间-雨滴大小-数密度", "mm${^-}$${^1}$·m${^-}$${^3}$"),
    'diameter_height_density': ("雨滴大小-高度-数密度", "mm${^-}$${^1}$·m${^-}$${^3}$")
}
