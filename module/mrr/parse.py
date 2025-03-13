__author__ = 'panchuang'


# from utils.filter_data import filter_by_time
from datetime import datetime


class ParseMRR:
    def __init__(self):
        self.times = []

    def QC_after(self, dat_path, time_range):
        data_list = []
        data_dict = {}
        d62 = []  # 雨滴直径
        n62 = []  # 雨滴数密度
        huibo = []
        zdr = []

        dindexs = [" D" + str(i).rjust(2, '0') for i in range(64)]
        nindexs = [" N" + str(i).rjust(2, '0') for i in range(64)]

        with open(dat_path, 'r') as f:
            for l in f.readlines():
                if l[:4] == " MRR":
                    val = l[4:]
                    time_list = val.split()
                    tim = "".join([i.rjust(2, '0') for i in time_list])
                    data_dict["01"] = str(datetime.now().year)[:2]+tim
                    self.times.append(tim)
                elif l[:4] == " H  ":
                    val = l[2:]
                    data_dict['02'] = val.split()
                elif l[:4] in dindexs:
                    val = l[4:]
                    if l[:4] != ' D63':
                        d62.append(val.split())
                    else:
                        d62.append(val.split())
                        data_dict['03'] = d62
                        d62 = []
                elif l[:4] in nindexs:
                    val = l[4:]
                    if l[:4] != ' N63':
                        n62.append(val.split())
                    else:
                        n62.append(val.split())
                        data_dict['04'] = n62
                        n62 = []
                elif l[:4] == " z  ":
                    val = l[2:]
                    data_dict['05'] = val.split()
                elif l[:4] == " Z  ":
                    val = l[2:]
                    data_dict['06'] = val.split()
                elif l[:4] == " RR ":
                    val = l[3:]
                    data_dict['07'] = [i if "*" not in i else "" for i in val.split()]

                elif l[:4] == " LWC":
                    val = l[4:]
                    data_dict['08'] = [i if "*" not in i else "" for i in val.split()]

                elif l[:4] == " W  ":
                    val = l[2:]
                    data_dict['09'] = val.split()
                elif l[:4] == " ZA ":
                    val = l[3:]
                    huibo.append(val.split())
                elif l[:4] == " ZU ":
                    val = l[3:]
                    huibo.append(val.split())
                elif l[:4] == " ZX ":
                    val = l[3:]
                    huibo.append(val.split())
                elif l[:4] == " ZC ":
                    val = l[3:]
                    huibo.append(val.split())
                elif l[:4] == " ZS ":
                    val = l[3:]
                    huibo.append(val.split())
                    data_dict['10'] = huibo  # 回波
                    huibo = []
                elif l[:4] == " DRA":
                    val = l[4:]
                    zdr.append(val.split())
                elif l[:4] == " DRU":
                    val = l[4:]
                    zdr.append(val.split())
                elif l[:4] == " DRX":
                    val = l[4:]
                    zdr.append(val.split())
                elif l[:4] == " DRC":
                    val = l[4:]
                    zdr.append(val.split())
                elif l[:4] == " DRS":
                    val = l[4:]
                    zdr.append(val.split())
                    data_dict['11'] = zdr  # ZDR
                elif l[:4] == " Nw ":
                    val = l[3:]
                    data_dict['12'] = val.split()
                elif l[:4] == " Dm ":
                    val = l[3:]
                    data_dict['13'] = val.split()
                elif l[:4] == " Mu ":
                    val = l[3:]
                    data_dict['14'] = val.split()
                    data_list.append(data_dict)
                    data_dict = {}
                else:
                    print(f"有问题的: head: {l[:4]}, all: {l}")

        # start_time, end_time = time_range.split(",")
        # data_list, tims = filter_by_time(data_list, [start_time[2:], end_time[2:]])
        # self.times = tims

        return data_list

    def QC_before(self, ave_path, time_range):
        # # 对应质控后文件中解析出的时间
        # times_range = ['210705000002', '210705000102', '210705000201', '210705000302', '210705000402', '210705000502',
        #                '210705000601', '210705000702', '210705000802', '210705000902']

        data_list = []
        data_dict = {}
        d62 = []  # 雨滴直径
        n62 = []  # 雨滴数密度

        # if not self.times:
        #     self.QC_after(AFTER_PATH_MRR, time_range)  # 设计上存在问题

        dindexs = ["D" + str(i).rjust(2, '0') for i in range(64)]
        nindexs = ["N" + str(i).rjust(2, '0') for i in range(64)]

        with open(ave_path, 'r') as f:
            for l in f.readlines():
                if l[:3] == "MRR":
                    val = l[3:16]
                    data_dict["01"] = f"{datetime.strptime(val.strip(), '%y%m%d%H%M%S'):%Y%m%d%H%M%S}"
                if l[:3] == "H  ":
                    val = l[3:]
                    data_dict['02'] = self.get_val_by_line(val, 31, 7)
                elif l[:3] in dindexs:
                    val = l[3:]
                    if l[:3] != 'D63':
                        d62.append(self.get_val_by_line(val, 31, 7))
                    else:
                        d62.append(self.get_val_by_line(val, 31, 7))
                        data_dict['03'] = d62
                        d62 = []
                elif l[:3] in nindexs:
                    val = l[3:]
                    if l[:3] != 'N63':
                        n62.append(self.get_val_by_line(val, 31, 7))
                    else:
                        n62.append(self.get_val_by_line(val, 31, 7))
                        data_dict['04'] = n62  # 没除1000
                        n62 = []
                elif l[:3] == "z  ":
                    val = l[3:]
                    data_dict['05'] = self.get_val_by_line(val, 31, 7)
                elif l[:3] == "Z  ":
                    val = l[3:]
                    data_dict['06'] = self.get_val_by_line(val, 31, 7)
                elif l[:3] == "RR ":
                    val = l[3:]
                    data_dict['07'] = self.get_val_by_line(val, 31, 7)
                elif l[:3] == "LWC":
                    val = l[3:]
                    data_dict['08'] = self.get_val_by_line(val, 31, 7)
                elif l[:3] == "W  ":
                    val = l[3:]
                    data_dict['09'] = self.get_val_by_line(val, 31, 7)
                    data_list.append(data_dict)
                    data_dict = {}

        # start_time, end_time = time_range.split(",")
        # data_list, tims = filter_by_time(data_list, [start_time[2:], end_time[2:]], self.times)

        return data_list

    def get_val_by_line(self, val, num, step):
        """
        获取每一行中的数据值
        :param val: str 行数据
        :param num: int 需要获取多少个数据，每种数据是固定的
        :param step: 每个值的长度
        :return:
        """
        start_index = 0
        H_v = []
        for i in range(step, (num + 1) * step, step):
            H_v.append(val[start_index:i].strip())
            start_index = i
        return H_v
