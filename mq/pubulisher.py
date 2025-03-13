#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from datetime import datetime

import requests
from loguru import logger
from pika import BlockingConnection, URLParameters

from config.config import config_info
mq_cfg = config_info.get_rabbitmq_cfg


def publish(msg):
    # 发送消息
    msg = json.dumps(msg)
    msg = msg.replace("NaN", "null").replace("nan", "null")    # 替换NaN值
    logger.info(f"===>{msg}")
    connection = BlockingConnection(URLParameters(mq_cfg['mq_url']))
    channel = connection.channel()

    channel.exchange_declare(exchange=mq_cfg['send_exchange'], exchange_type='fanout')
    channel.basic_publish(exchange=mq_cfg['send_exchange'], routing_key='', body=msg)
    connection.close()


def call_service_api(payload):
    payload = json.dumps(payload)
    payload = payload.replace("NaN", "null").replace("nan", "null")
    # service_name = 'cl-arithmetic-service'
    # endpoint = '/arithmetic/handleResult'
    # headers = {'Content-Type': 'application/json'}
    # service_instances_url = f'http://nacos.clizard.team:8848/nacos/v1/ns/instance/list?serviceName={service_name}&&namespaceId=gzfb-project&&groupName=uat'
    # response = requests.get(service_instances_url)
    # instances = response.json()['hosts']
    # if not instances:
    #     raise Exception(f"No instances available for service: {service_name}")

    # instance = instances[0]  # 使用第一个实例，你可以根据需求进行选择
    logger.info(f"===>{payload}")
    server_info = config_info.get_data_server_cfg
    headers = server_info['headers']
    service_url = server_info['service_url']
    response = requests.post(service_url, data=payload, headers=headers)
    logger.info(response.json())
    # if response.json()['success'] is False:
    #     raise Exception(response.json()['msg'])


def publish_start_msg(job_id):
    # 任务开始消息
    msg = {
        "data": {
            "status": "10",
            "jobId": job_id,
            "noticeType": 1,
            "result": {
                "resultData": "执行中"
            },
            "startTime": str(datetime.now())[:19]
        }
    }
    call_service_api(msg)


def publish_complete_msg(status_code, job_id, image_name, result_data):
    # 任务完成消息
    msg = {
        "data": {
            "status": status_code,
            "jobId": job_id,
            "noticeType": 1,
            "imageName": image_name,
            "result": {
                "resultData": result_data
            },
            "endTime": str(datetime.now())[:19]
        }
    }
    call_service_api(msg)


def publish_failed_msg(image_name, job_id, e):
    # 任务失败消息
    msg = {
        "data": {
            "status": "-1",
            "imageName": image_name,
            "jobId": job_id,
            "noticeType": 1,
            "result": f"任务执行失败, 错误信息:（{e}）",
            "startTime": str(datetime.now())[:19]
            }
    }

    call_service_api(msg)


def publish_temp_msg(status_code, job_id, image_name, result_data):
    # 中间文件推送消息
    msg = {
        "data": {
            "status": status_code,
            "jobId": job_id,
            "noticeType": 1,
            "fileType": "temp",
            "imageName": image_name,
            "result": {
                "resultData": result_data
            },
            "endTime": str(datetime.now())[:19]
        }
    }
    call_service_api(msg)
