"""
mq客户端，实现生产和消费消息功能
"""

import json
import threading
import time
from typing import Callable, Optional

import pika
from pika.exceptions import AMQPConnectionError
from loguru import logger

from service.handler import handle_msg

from config.config import config_info


class RabbitMQ(object):
    def __init__(self):
        # connect to mq
        self.mq_cfg = config_info.get_rabbitmq_cfg
        self._exchange = self.mq_cfg['exchange']

        self._connection, self._channel = self._connect_mq(self.mq_cfg['mq_url'])

    def _connect_mq(self, mq_url):
        # 连接mq
        try:
            connection = pika.BlockingConnection(pika.URLParameters(mq_url))
            channel = connection.channel()
            channel.exchange_declare(
                exchange=self._exchange,
                exchange_type='fanout',
                durable=False
            )
            logger.info('RabbitMQ: Connection and Channel Created.')
            return connection, channel
        except Exception as e:
            logger.error(f'RabbitMQ Connect failed: {e}')
            logger.info('Retry in 3 seconds...')
            time.sleep(3)
            return self._connect_mq(self.mq_cfg['mq_url'])

    def __del__(self):
        # 关闭mq连接
        logger.info("close connection")
        print("end", time.time())
        self._connection.close()

    def publish(
            self, routing_key: str, message: bytes, status='RUNNING',
            expiration=3600, failed_then: Optional[Callable] = None, queue_count=1
    ) -> None:
        """Publish a message to RabbitMQ exchange.

        param str routing_key: The routing key to build on.
        param bytes message: The message body; empty string if no body
        :param queue_count:
        :param expiration:
        :param status:
        :param message:
        :param routing_key:
        :param callable failed_then: callback function, when publish failed will call callback(message)
        """
        # create a new connection
        connection, channel = self._connect_mq(self.mq_cfg['mq_url'])
        # enable confirm
        channel.confirm_delivery()
        # publish message
        try:
            channel.basic_publish(
                exchange=self._exchange,
                routing_key=routing_key,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,
                ),
                mandatory=True
            )
            logger.info('Message sent successfully')
        except Exception as e:
            logger.error(f'Message sent failed: {e}')
            if failed_then:
                # callback
                failed_then(message)
        logger.info('Sent %r: %r' % (routing_key, message))
        # close connection
        connection.close()

    def _ack_message(self, ch, delivery_tag) -> None:
        """Note that `ch` must be the same pika channel instance via which
        the message being ACKed was retrieved (AMQP protocol constraint).
        """
        if ch.is_open:
            ch.basic_ack(delivery_tag)
        else:
            # Channel is already closed, so we can't ACK this message;
            # log and/or do something that makes sense for your app in this case.
            pass

    def on_message(self, channel, method, properties, body):
        load_msg = json.loads(body.decode())
        message_thread = threading.Thread(target=handle_msg, args=(load_msg,))
        message_thread.start()
        while message_thread.is_alive():
            time.sleep(1)
            self._connection.process_data_events()

        channel.basic_ack(delivery_tag=method.delivery_tag)

    def start_consuming(self, queue, ttl_minutes=3600):
        """Consumers start to consume, using this method will start to get content from the message queue and process it.

        params str queue: Queue name.
        params str|list binding_keys: A list containing multiple binding keys.
            If only one key is bound, a string can also be passed in.
        params int ttl_second: Message expiration time.
        params callable callback: The callback function that needs to be executed.
            Default method will only print the key and body of the message, you need to write this function by yourself.
            The format of the callback function is like this:

                def callback(message_body):
                    pass
        """
        ttl = ttl_minutes * 1000 * 24
        result = self._channel.queue_declare(
            queue, durable=False, arguments={'x-expires': ttl})
        queue_name = result.method.queue

        self._channel.queue_bind(exchange=self._exchange, queue=queue_name, routing_key='')

        logger.info('Waiting for task. To exit press CTRL+C')

        self._channel.basic_qos(prefetch_count=1)
        self._channel.basic_consume(queue=queue_name, on_message_callback=self.on_message)

        try:
            self._channel.start_consuming()
        except KeyboardInterrupt:
            # stop consuming
            self._channel.stop_consuming()
        except AMQPConnectionError:
            # reconnect to mq
            logger.info('Retry in 3 seconds...')
            time.sleep(3)
            self._connection, self._channel = self._connect_mq(self.mq_cfg['mq_url'])
            # start again
            self.start_consuming(queue, ttl_minutes)
        except Exception as e:
            logger.info(f"get message error: {e}")
            logger.info('stop consuming...')
