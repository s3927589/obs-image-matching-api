from fastapi import FastAPI, status, File, UploadFile, BackgroundTasks
import numpy as np
import time
import cv2
import json
import os
from models import AddItemForm, DeleteItemForm
from fastapi.middleware.cors import CORSMiddleware
from classifier import Classifier
import uuid
import logging


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./obs-rmit-686bb94f09ee.json"
logging.basicConfig(level=logging.INFO)
app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "efficient.onnx"
CLF_PATH = "knn.pickle"
CLASS_PATH = "classes.json"

STORAGE_DIR = "ai/"
MY_SECRET_KEY = "bdaba086-70a2-4034-a64b-fedf01677751"

model = Classifier(MODEL_PATH, CLF_PATH, CLASS_PATH, STORAGE_DIR)
queue_task = []
timeout = 3600


def create_payload(method, data):
    payload = {
        "method": method,
        "data": data
    }

    return json.dumps(payload)


def extract_payload(payload_str):
    payload = json.loads(payload_str)

    return payload["method"], payload["data"]


def my_task(method, data):
    global queue_task
    my_id = str(uuid.uuid4())
    logging.info(f"Adding {method} {my_id} to the queue")
    queue_task.append(my_id)

    try:
        st = time.time()
        while True:
            if time.time() - st > timeout:
                break

            if len(queue_task) == 0:
                logging.info("No task to do")
                return

            if queue_task[0] == my_id:
                break
            logging.info(f"Check {method} {my_id}")
            time.sleep(5)
            continue
    except:
        logging.info("Error when processing queue")
        return

    logging.info(f"Processing {method} {my_id}")
    try:
        if method == "add":
            add_item_task(*data)
        elif method == "delete":
            delete_item_task(*data)
        elif method == "update":
            update_item_task(*data)
    except:
        logging.info(f"Error {method} {my_id}")

    # the task is done, remove it from the queue
    queue_task.pop(0)


def add_item_task(item_id, imageUrls):
    logging.info("=====================================")
    logging.info(f"Start ADDING task {item_id}")
    res = model.add_item(item_id, imageUrls)
    logging.info(f"End ADDING task {item_id} {res}")
    logging.info("=====================================")


def delete_item_task(item_id):
    logging.info("=====================================")
    logging.info(f"Start DELETING task {item_id}")
    res = model.delete_item(item_id)
    logging.info(f"End DELETING task {item_id} {res}")
    logging.info("=====================================")


def update_item_task(item_id, imageUrls):
    logging.info("=====================================")
    logging.info(f"Start UPDATING task {item_id}")
    res = model.update_item(item_id, imageUrls)
    logging.info(f"End UPDATING task {item_id} {res}")
    logging.info("=====================================")


@app.get('/', status_code=status.HTTP_200_OK)
def hello():
    return {
        "message": "Welcome to Minh Anh's AI world"
    }


@app.get('/api/classes', status_code=status.HTTP_200_OK)
def get_classes():
    if model is not None:
        return [model.class_dict, model.class_dict_reversed]

    return {
        "message": "Class not found"
    }


# @app.get('/api/reset', status_code=status.HTTP_200_OK)
# def reset_model():
#     model.load_data()

#     return {
#         "message": "Model is reset successfully"
#     }


@app.post('/api/image', status_code=status.HTTP_200_OK)
async def compare_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = model.get_pred(img)
        logging.info(f"Result {result}")

        return {
            "itemIds": result
        }
    except:
        pass

    return {
        "itemIds": []
    }


@app.post('/api/add_item', status_code=status.HTTP_200_OK)
async def add_item(data: AddItemForm, background_tasks: BackgroundTasks):
    item_id = data.item_id
    secret_key = data.secret_key
    imageUrls = data.imageUrls

    if secret_key != MY_SECRET_KEY:
        return {
            "message": "Failed to authenticate",
        }

    background_tasks.add_task(my_task, "add", [item_id, imageUrls])
    return {
        "message": "Adding new item",
    }


@app.post('/api/delete_item', status_code=status.HTTP_200_OK)
async def delete_item(data: DeleteItemForm, background_tasks: BackgroundTasks):
    item_id = data.item_id
    secret_key = data.secret_key

    if secret_key != MY_SECRET_KEY:
        return {
            "message": "Failed to authenticate",
        }

    background_tasks.add_task(my_task, "delete", [item_id])
    return {
        "message": "Deleting new item",
    }


@app.put('/api/update_item', status_code=status.HTTP_200_OK)
async def update_item(data: AddItemForm, background_tasks: BackgroundTasks):
    item_id = data.item_id
    secret_key = data.secret_key
    imageUrls = data.imageUrls

    if secret_key != MY_SECRET_KEY:
        return {
            "message": "Failed to authenticate",
        }

    background_tasks.add_task(my_task, "update", [item_id, imageUrls])
    return {
        "message": "Updating new item",
    }
