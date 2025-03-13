import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ele_name_list = ['时间', '气压', '温度', '湿度', '露点', '位势高度', '风速', '风向', '经度', '纬度', '海拔',
#        '距离', '升速', '方位角', '仰角']
from config.config import config_info
from module.public.tlnp import tlnp
from service.utils import transfer_path, file_exit
# from utils.file_uploader import minio_client

lin_info = {'tem': '℃',
            'pre': 'hPa',
            'rh': '%',
            'wd': '°',
            'ws': 'm/s', }


class GPSliners:
    def __init__(self, input_filelist, ele, title):
        self.file_names = file_exit(input_filelist[0].get('filePath'))
        self.ele = ele
        self.title = title
        self.pic_path = config_info.get_gpstk_cfg['pic_path']

    def get_pro_data(self):
        with open(self.file_names, 'r') as f:
            content_list = [i.strip() for i in f]
            station_id = content_list[0].split(' ')[1]
            ob_time = datetime.strptime(content_list[5].split(': ')[1], "%Y-%m-%d %H:%M:%S")  # 观测时间
        with open(self.file_names, 'r') as f:
            df = pd.read_csv(f, sep='\s+', skiprows=13)
            df.columns = [i.split('[')[0] for i in df.columns.values]
            ws = df.风速.tolist()
            wd = df.风向.tolist()
            ups = df.升速.tolist()
            tem = df.温度.tolist()
            pre = df.气压.tolist()
            rh = df.湿度.tolist()
            td = df.露点.tolist()
            h = df.海拔 / 1000
            hz = df.位势高度.tolist()
            h = np.around(h, 2).tolist()

        data = {'tem': tem, 'pre': pre, 'rh': rh, 'wd': wd, 'ws': ws, 'td': td, 'ups': ups,
                'h': h, 'hz': hz, 't': ob_time, 'id': station_id}
        return data

    def draw_tlnp(self, data):
        fig, _ = tlnp(data['pre'], data['tem'], data['td'], data['hz'], data['ws'], data['wd'],
                      up_speed=data['ups'], station_id=data['id'], valid_time=data['t'])
        pic_path = os.path.join(self.pic_path, data['t'].strftime('%Y%m'))
        pic_name = f"tlnp_{data['t'].strftime('%Y%m%d%H%M')}_t.png"
        if not Path(pic_path).exists():
            Path.mkdir(Path(pic_path), parents=True, exist_ok=True)
        pic_path = os.path.join(pic_path, pic_name)
        fig.savefig(pic_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        # minio_client.put_object(pic_path)
        pic_info = {"filename": os.path.basename(pic_path), "path": transfer_path(pic_path, is_win_path=True),
                    "element": self.ele}
        return pic_info

    def run(self):
        data = self.get_pro_data()
        line_data = []
        pic_info = []
        if self.ele == 'tlnp':
            pic_info.append(self.draw_tlnp(data))
            return [{"picFiles": pic_info, "picData": line_data}]

        line_data = [{
            'x': data[self.ele][::60],  # 将数据取为分钟级
            'y': [data['h'][::60]],  #
            'xlabel': lin_info[self.ele],
            'ylabel': 'km',
            'yname': [self.title],
            'time': data['t'].strftime('%Y%m%d%H%M')
        }]
        return [{"picFiles": pic_info, "picData": line_data}]

# if __name__ == '__main__':
#     file = [{"filePath": "E:/data/探空/GPS探空/2022080113/EDT_20220801_13.txt"}]
#     ele = 'tlnp'
#     title = 'test'
#     print(GPSliners(file, ele, title).run())
