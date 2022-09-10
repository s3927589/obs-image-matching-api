from fastapi import FastAPI, status, File, UploadFile, BackgroundTasks
import numpy as np
import cv2
import json
from models import AddItemForm
from fastapi.middleware.cors import CORSMiddleware
from classifier import Classifier
from rabbitmq_provider import Producer


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
producer = Producer("localhost", "5672")


def create_payload(method, data):
    payload = {
        "method": method,
        "data": data
    }

    return json.dumps(payload)


def my_task(method, data):
    if method == "add":
        add_item_task(*data)
    elif method == "delete":
        delete_item_task(*data)
    elif method == "update":
        update_item_task(*data)


def add_item_task(item_id, imageUrls):
    print("Start adding tasks", item_id)
    res = model.add_item(item_id, imageUrls)
    print("End adding tasks", item_id, res)


def delete_item_task(item_id):
    print("Start deleting tasks", item_id)
    res = model.delete_item(item_id)
    print("End deleting tasks", item_id, res)


def update_item_task(item_id, imageUrls):
    print("Start deleting tasks", item_id)
    res = model.update_item(imageUrls)
    print("End deleting tasks", item_id, res)


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
# async def add_item(data: AddItemForm, background_tasks: BackgroundTasks):
async def add_item(data: AddItemForm):
    item_id = data.item_id
    imageUrls = data.imageUrls

    producer.send_data(create_payload("add", [item_id, imageUrls]))

    # background_tasks.add_task(my_task, "add", [item_id, imageUrls])
    return {
        "message": "Adding new item",
    }


@app.delete('/api/delete_item/{item_id}', status_code=status.HTTP_200_OK)
# async def delete_item(item_id: str, background_tasks: BackgroundTasks):
async def delete_item(item_id: str):
    # background_tasks.add_task(my_task, "delete", [item_id])
    producer.send_data(create_payload("delete", [item_id]))
    return {
        "message": "Deleting new item",
    }


@app.put('/api/update_item', status_code=status.HTTP_200_OK)
# async def update_item(data: AddItemForm, background_tasks: BackgroundTasks):
async def update_item(data: AddItemForm):
    item_id = data.item_id
    imageUrls = data.imageUrls

    # background_tasks.add_task(my_task, "update", [item_id, imageUrls])
    producer.send_data(create_payload("update", [item_id, imageUrls]))
    return {
        "message": "Updating new item",
    }
