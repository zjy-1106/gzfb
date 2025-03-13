__author__ = 'panchuang'

from datetime import datetime
# from utils.filter_data import filter_by_time


class ParseDSD:
    def __init__(self):
        self.times = []  # 存放质控后的时间

    def QC_after(self, data_path, time_range):
        data_list = []
        data_dict = {}
        for data_p in data_path:
            with open(data_p, 'r') as f:
                for l in f.readlines():
                    head = l[:4][:2]
                    val = l[4:]
                    if head == '01':
                        time_list = val.split()
                        tim = "".join([i.rjust(2, '0') for i in time_list])
                        data_dict[head] = tim + '00'   # 返回时间精确到秒
                        self.times.append(tim)
                    elif head == '02':
                        data_dict[head] = val.split()
                    elif head == '03':
                        data_dict[head] = val.split()
                    elif head == '04':
                        data_dict[head] = val.split()
                    elif head == '05':
                        data_dict[head] = val.split()
                    elif head == '06':
                        data_dict[head] = val.split()
                    elif head == '07':
                        data_dict[head] = val.split()
                    elif head == '08':
                        data_dict[head] = val.split()
                    elif head == '09':
                        data_dict[head] = val.split()
                    elif head == '10':
                        data_dict[head] = val.split()
                    elif head == '11':
                        data_dict[head] = val.split()
                    elif head == '12':
                        data_dict[head] = val.split()
                        data_list.append(data_dict)
                        data_dict = {}
                    # else:
                    #     print(f"有问题的: head: {head}, all: {l}")
                # print(data_list)
                # print(times)

        # start_time, end_time = time_range.split(',')
        # data_list, tims = filter_by_time(data_list, [start_time[:-2], end_time[:-2]])
        # self.times = tims

        return data_list

    def QC_before(self, txt_path, time_range, AFTER_PATH_DSD):
        data_list = []

        if not self.times:
            self.QC_after(AFTER_PATH_DSD, time_range)  # 设计上存在问题

        #
        # times_range = ['202110301657', '202110310126', '202110310127', '202110310128', '202110310129', '202110310136',
        #                '202110310137', '202110310138', '202110310139', '202110310140', '202110310141', '202110310142',
        #                '202110310143', '202110310144', '202110310145', '202110310254', '202110310255']
        for txt_p in txt_path:
            with open(txt_p, 'r', encoding='gbk') as f:
                while 1:
                    data = []
                    data_dict = {}
                    i = 0
                    for line in f:
                        data.append(line.strip())
                        i = i + 1
                        if i == 88:
                            break
                    if len(data) < 88:
                        break

                    tim = datetime.strptime(f'{data[20][3:]} {data[19][3:]}', '%d.%m.%Y %H:%M:%S').strftime('%Y%m%d%H%M%S')

                    # if times_range[0]<=tim<=times_range[1]:从时间段内筛选
                    if tim in self.times:
                        data_dict['01'] = tim
                        data_dict['07'] = [data[0][3:], data[1][3:], 0, 0, 0]  # 0无特殊含义，用来占位
                        data_dict['06'] = data[6][3:]
                        data_list.append(data_dict)
                    else:
                        pass
        # start_time, end_time = time_range.split(',')
        # data_list, tims = filter_by_time(data_list, [start_time[:-2], end_time[:-2]])
        return data_list
