# -*- coding: utf-8 -*-
"""
MRR微雨雷达服务入口，监听消息队列，将雷达原始文件转化为nc，绘制产品图
"""
import os
from loguru import logger

print(os.getcwd())
os.environ['BASEMAPDATA'] = os.getcwd()

from mq.blocking_consumer import RabbitMQ
from config.config import config_info

if __name__ == '__main__':
    logger.info("开始监听...")
    mq_cfg = config_info.get_rabbitmq_cfg

    mq = RabbitMQ()
    mq.start_consuming(mq_cfg['queue_name'])
    # consumer = ReconnectConsumer(mq_cfg['mq_url'], mq_cfg['queue_name'])
    # consumer.run()
