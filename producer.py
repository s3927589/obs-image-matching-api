import pika
import json

credentials = pika.PlainCredentials("guest", "guest")
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost',
                              port="5672",
                              credentials=credentials))
channel = connection.channel()

channel.queue_declare(queue='obs')

while True:
    text = input("Message:")
    data = {
        "id": 0,
        "data": ["hello", text]
    }
    payload = json.dumps(data)
    channel.basic_publish(exchange='', routing_key='hello', body=payload)
    print(f" [x] Sent '{text}'")

connection.close()
