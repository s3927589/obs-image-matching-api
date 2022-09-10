import pika
import json
import requests


class Producer:
    def __init__(self, host, port):
        credentials = pika.PlainCredentials("guest", "guest")
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host,
                                      port=port,
                                      credentials=credentials))
        self.channel = self.connection.channel()

        self.channel.queue_declare(queue='obs')

    def send_data(self, payload):
        self.channel.basic_publish(exchange='',
                                   routing_key='obs',
                                   body=payload)
        print(f"[RabbitMQ Producer] Sent '{payload}'")

    def close(self):
        self.connection.close()


class Consumer:
    def __init__(self, host, port, model, reset_url):
        credentials = pika.PlainCredentials("guest", "guest")
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host,
                                      port=port,
                                      credentials=credentials))

        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='obs')

        self.channel.basic_consume(queue='obs',
                                   on_message_callback=self.callback,
                                   auto_ack=True)
        self.model = model
        self.reset_url = reset_url

    def callback(self, ch, method, properties, body):
        payload = json.loads(body.decode('utf-8'))
        method = payload["method"]
        data = payload["data"]
        print("Received", payload)

        if method == "add":
            self.add_item_task(*data)
        elif method == "delete":
            self.delete_item_task(*data)
        elif method == "update":
            self.update_item_task(*data)

        # reset model in the main API
        requests.get(url=self.reset_url)

    def add_item_task(self, item_id, imageUrls):
        print("=====================================")
        print("Start ADDING task", item_id)
        res = self.model.add_item(item_id, imageUrls)
        print("End ADDING task", item_id, res)
        print("=====================================")

    def delete_item_task(self, item_id):
        print("=====================================")
        print("Start DELETING task", item_id)
        res = self.model.delete_item(item_id)
        print("End DELETING task", item_id, res)
        print("=====================================")

    def update_item_task(self, item_id, imageUrls):
        print("=====================================")
        print("Start UPDATING task", item_id)
        res = self.model.update_item(item_id, imageUrls)
        print("End UPDATING task", item_id, res)
        print("=====================================")

    def start(self):
        self.channel.start_consuming()
