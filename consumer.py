import sys
import os
from rabbitmq_provider import Consumer
from classifier import Classifier


def main():
    MODEL_PATH = "efficient.onnx"
    CLF_PATH = "knn.pickle"
    CLASS_PATH = "classes.json"
    STORAGE_DIR = "ai/"

    try:
        PORT = os.getenv('PORT')
        if PORT is None:
            PORT = "8000"
            os.environ["PORT"] = PORT
    except:
        PORT = "8000"
        os.environ["PORT"] = PORT
        pass

    reset_url = f"http://localhost:{PORT}/api/reset"

    model = Classifier(MODEL_PATH, CLF_PATH, CLASS_PATH, STORAGE_DIR)
    consumer = Consumer("localhost", "5672", model, reset_url)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    consumer.start()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
