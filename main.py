from fastapi import FastAPI, status, File, UploadFile, BackgroundTasks
import numpy as np
import time
import cv2
import json
from models import AddItemForm
from fastapi.middleware.cors import CORSMiddleware
from classifier import Classifier
import uuid


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

model = Classifier(MODEL_PATH, CLF_PATH, CLASS_PATH, STORAGE_DIR)
queue_task = []


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
    my_id = uuid.uuid4()
    print(f"Adding {method} {my_id} to the queue")
    queue_task.append(my_id)

    try:
        while True:
            if queue_task[0] == my_id:
                break
            print("Check", method, my_id)
            time.sleep(5)
            continue
    except:
        print("Error when processing queue")
        return

    print(f"Processing {method} {my_id}")
    if method == "add":
        add_item_task(*data)
    elif method == "delete":
        delete_item_task(*data)
    elif method == "update":
        update_item_task(*data)

    # the task is done, remove it from the queue
    queue_task.pop(0)


def add_item_task(item_id, imageUrls):
    print("=====================================")
    print("Start ADDING task", item_id)
    res = model.add_item(item_id, imageUrls)
    print("End ADDING task", item_id, res)
    print("=====================================")


def delete_item_task(item_id):
    print("=====================================")
    print("Start DELETING task", item_id)
    res = model.delete_item(item_id)
    print("End DELETING task", item_id, res)
    print("=====================================")


def update_item_task(item_id, imageUrls):
    print("=====================================")
    print("Start UPDATING task", item_id)
    res = model.update_item(item_id, imageUrls)
    print("End UPDATING task", item_id, res)
    print("=====================================")


@app.get('/api/classes', status_code=status.HTTP_200_OK)
def get_classes():
    if model is not None:
        return [model.class_dict, model.class_dict_reversed]

    return {
        "message": "Class not found"
    }


@app.get('/api/reset', status_code=status.HTTP_200_OK)
def reset_model():
    model.load_data()

    return {
        "message": "Model is reset successfully"
    }


@app.post('/api/image', status_code=status.HTTP_200_OK)
async def compare_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = model.get_pred(img)
        print("Result", result)

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
    imageUrls = data.imageUrls

    background_tasks.add_task(my_task, "add", [item_id, imageUrls])
    return {
        "message": "Adding new item",
    }


@app.delete('/api/delete_item/{item_id}', status_code=status.HTTP_200_OK)
async def delete_item(item_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(my_task, "delete", [item_id])
    return {
        "message": "Deleting new item",
    }


@app.put('/api/update_item', status_code=status.HTTP_200_OK)
async def update_item(data: AddItemForm, background_tasks: BackgroundTasks):
    item_id = data.item_id
    imageUrls = data.imageUrls

    background_tasks.add_task(my_task, "update", [item_id, imageUrls])
    return {
        "message": "Updating new item",
    }
