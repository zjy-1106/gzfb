#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

from loguru import logger

from config.config import config_info
from service.utils import upload_files, del_rmdir


def upload_star():
    path_cfg = config_info.get_path
    mid_path = ['mrr\data', 'dsd\data', 'rainrate\data', 'dda_wind\data', 'cwr\data', 'mwr\data', 'cr\data']  #
    try:
        upload_files(mid_path)
    except PermissionError:
        logger.info('文件使用中，等待20秒')
        time.sleep(20)
        upload_star()
    except Exception as e:
        logger.info(f"上传错误: {e}")

    # del_rmdir(path_cfg['file_save_path'])  # 删除目录下所有空文件夹


if __name__ == '__main__':
    T = True
    while T:
        upload_star()
