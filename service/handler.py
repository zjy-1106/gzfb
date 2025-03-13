#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析消息体，绘制产品图
"""
import json
import traceback


from mq.pubulisher import publish_start_msg, publish_failed_msg, mq_cfg
from service.message_dispose import *

# pool = ThreadPoolExecutor(5)


def handle_msg(msg):
    # 解析消息体
    message = msg['data']
    jobid = message['jobId']
    if message['imageName'] not in mq_cfg['imageName']:  # 只处理当前算法的消息
        return

    radar_type = message['imageName']
    try:
        # 调用初始程序
        req = Message()
        req.image_name = message['imageName']

        req.requestContent = message
        logger.info(json.dumps(message, ensure_ascii=False))
        publish_start_msg(jobid)  # 发送开始消息
        observe = eval(f'{radar_type}()')
        observe.handle_request(req)  # 调用初始程序

    except Exception as e:
        # 发送执行失败通知
        traceback.print_exc()
        publish_failed_msg(message['imageName'], jobid, e)
        logger.error(e)


# def handle_msg(msg):
#     # 解析消息体
#     pool.submit(handle_msg_by_thread, msg)

