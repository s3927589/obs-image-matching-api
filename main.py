from fastapi import FastAPI, status, File, UploadFile
import pickle
import os
import json
import numpy as np
import urllib.request
import random
from tqdm import tqdm
import onnxruntime as rt
import cv2
from sklearn.neighbors import KNeighborsClassifier
from models import AddItemForm
import pyrebase


config = {
    "apiKey": "AIzaSyDHDyFoXS-E7sZNJQ9G9Px9AEMStUtyrjw",
    "authDomain": "obs-rmit.firebaseapp.com",
    "databaseURL": "",
    "projectId": "obs-rmit",
    "storageBucket": "obs-rmit.appspot.com",
    "serviceAccount": "./obs-rmit-firebase-adminsdk.json"
}

app = pyrebase.initialize_app(config)
storage = app.storage()

# storage.child("ai/knn.pickle").put("../api/knn.pickle")
# storage.download("ai/knn.pickle", "knn.pickle")


app = FastAPI()
TARGET_SIZE = (300, 300)
MODEL_PATH = "efficient.onnx"
CLF_PATH = "knn.pickle"
CLASS_PATH = "classes.json"
# X_TRAIN_PATH = "data.npy"
# Y_TRAIN_PATH = "labels.npy"
data_path_list = [MODEL_PATH, CLF_PATH, CLASS_PATH]

STORAGE_DIR = "ai/"

for data_path in data_path_list:
    if not os.path.isfile(data_path):
        print("Downloading", data_path)
        storage.child(STORAGE_DIR + data_path).download(STORAGE_DIR + data_path, "./" + data_path)
        print("Downloaded", data_path)


# model = load_model("./efficient.h5")
providers = ['CPUExecutionProvider']
model = rt.InferenceSession(MODEL_PATH, providers=providers)


clf = None
# x_train = np.array([])
# y_train = np.array([])
class_dict = {}
class_dict_reversed = {}

try:
    with open(CLF_PATH, "rb") as f:
        clf = pickle.load(f)

    # with open(X_TRAIN_PATH, "rb") as f:
    #     x_train = pickle.load(f)

    # with open(Y_TRAIN_PATH, "rb") as f:
    #     y_train = pickle.load(f)

    with open(CLASS_PATH, "r") as f:
        classes = json.load(f)
        class_dict = classes["class_dict"]
        class_dict_reversed = classes["class_dict_reversed"]
except:
    print("File data not found")


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    return img


def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    return img


def load_img_from_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def get_feat(img):
    return model.run(["top_dropout"],
                     {"input_4": np.expand_dims(img, axis=0).astype("float32")})[0][0]
    # return model.predict(np.expand_dims(img, axis=0))[0]


def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


def change_color(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    h[h > lim] = 255
    h[h <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img


def augment(img):
    imgFlipVer = np.flip(img, axis=0)
    imgFlipHor = np.flip(img, axis=1)
    imgBright1 = increase_brightness(img, random.randint(0, 50))
    imgBright2 = increase_brightness(img, random.randint(0, 50))
    imgFalseColor1 = change_color(img, random.randint(0, 50))
    imgFalseColor2 = change_color(imgFlipHor, random.randint(0, 50))
    imgFalseColor3 = change_color(imgFlipVer, random.randint(0, 50))
    imgFalseColor4 = change_color(img, random.randint(0, 50))
    imgFalseColor5 = change_color(imgFlipHor, random.randint(0, 50))
    imgFalseColor6 = change_color(imgFlipVer, random.randint(0, 50))
    imgGray = np.stack((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),)*3, axis=-1)
    imgGrayFlipVer = np.flip(imgGray, axis=0)
    imgGrayFlipHor = np.flip(imgGray, axis=1)

    return [img, imgFlipVer, imgFlipHor, imgBright1, imgBright2,
            imgFalseColor1, imgFalseColor2,
            imgFalseColor3, imgFalseColor4,
            imgFalseColor5, imgFalseColor6,
            imgGray,  imgGrayFlipVer, imgGrayFlipHor]


def extract_features(img_list):
    data = []
    for raw_img in tqdm(img_list):
        try:
            imgAugList = augment(raw_img)
            # for img in imgAugList:
            #     feat = get_feat(img)
            # feats = model.predict(np.array(imgAugList))
            feats = model.run(["top_dropout"],
                              {"input_4": np.array(imgAugList).astype("float32")})[0]
            data.extend(feats)
        except:
            continue
    return np.array(data)


def get_pred(img):
    feat = get_feat(img)
    dis, pred = clf.kneighbors(np.expand_dims(feat, axis=0), n_neighbors=4)
    dis = dis[0]
    pred = pred[0]
    filter_pred = []
    pred = pred[dis < 0.75]
    print(dis)
    print(pred)
    y_train = clf._y
    for index in pred:
        try:
            if y_train[index] not in filter_pred:
                filter_pred.append(y_train[index])
        except:
            pass

    result = []
    for index in filter_pred:
        ans = class_dict_reversed.get(str(index), None)
        if ans is not None:
            result.append(ans)
    return result


def update_storage():
    # save models
    with open(CLF_PATH, "wb") as f:
        pickle.dump(clf, f)
    # with open(X_TRAIN_PATH, "wb") as f:
    #     pickle.dump(x_train, f)
    # with open(Y_TRAIN_PATH, "wb") as f:
    #     pickle.dump(y_train, f)
    with open(CLASS_PATH, "w") as f:
        json.dump({
            "class_dict": class_dict,
            "class_dict_reversed": class_dict_reversed,
        }, f)

    for data_path in data_path_list[1:]:
        if os.path.isfile(data_path):
            print("Uploading", data_path)
            storage.child(STORAGE_DIR + data_path).put("./" + data_path)
            print("Uploaded", data_path)


@app.post('/api/image', status_code=status.HTTP_200_OK)
async def compare_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = preprocess(img)
        result = get_pred(img)

        return {
            "itemIds": result
        }
    except:
        pass

    return {
        "itemIds": []
    }


@app.post('/api/add_item', status_code=status.HTTP_200_OK)
async def add_item(data: AddItemForm):
    item_id = data.item_id
    imageUrls = data.imageUrls
    # global x_train, y_train, clf, class_dict, class_dict_reversed
    global clf, class_dict, class_dict_reversed
    x_train = clf._fit_X
    y_train = clf._y
    try:
        img_list = []
        for url in imageUrls:
            print(url)
            try:
                img = load_img_from_url(url)
            except:
                print("Cannot load image", url)
                continue
            print("Loaded")
            img = preprocess(img)
            # cv2.imwrite(f"Img {len(img_list)}.jpg", img)
            img_list.append(img)

        x = extract_features(img_list)
        new_label = int(np.max(y_train) + 1)
        y = [new_label] * len(x)
        y = np.array(y)

        x_train = np.concatenate([x_train, x])
        y_train = np.concatenate([y_train, y])
        print(y_train)

        # train the model
        clf_temp = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        clf_temp.fit(x_train, y_train)

        clf = clf_temp
        # x_train = x_train_temp
        # y_train = y_train_temp

        class_dict[item_id] = new_label
        class_dict_reversed[str(new_label)] = item_id

        update_storage()

        return {
            "success": True
        }
    except:
        pass

    return {
        "success": False
    }


@app.delete('/api/delete_item/{item_id}', status_code=status.HTTP_200_OK)
async def delete_item(item_id: str):
    # global x_train, y_train, clf, class_dict, class_dict_reversed
    global clf, class_dict, class_dict_reversed
    x_train = clf._fit_X
    y_train = clf._y

    if item_id not in class_dict:
        return {
            "success": False,
            "message": "Item does not exist"
        }

    target_label = int(class_dict[item_id])

    x_train = x_train[y_train != target_label]
    y_train = y_train[y_train != target_label]
    print('Target label', target_label)
    print(y_train)

    # train the model
    clf_temp = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    clf_temp.fit(x_train, y_train)

    clf = clf_temp
    # x_train = x_train_temp
    # y_train = y_train_temp

    class_dict = {x: y for x, y in class_dict.items() if x != item_id}
    class_dict_reversed = {x: y for x, y in class_dict_reversed.items() if y != item_id}

    # save models
    update_storage()

    return {
        "success": True,
        "message": "Deleted item successfully."
    }


@app.put('/api/update_item', status_code=status.HTTP_200_OK)
async def update_item(data: AddItemForm):
    item_id = data.item_id
    imageUrls = data.imageUrls
    # global x_train, y_train, clf, class_dict, class_dict_reversed
    global clf, class_dict, class_dict_reversed
    x_train = clf._fit_X
    y_train = clf._y

    # delete old data
    if item_id not in class_dict:
        return {
            "success": False,
            "message": "Item does not exist"
        }

    # retrain the models
    try:
        img_list = []
        for url in imageUrls:
            print(url)
            try:
                img = load_img_from_url(url)
            except:
                print("Cannot load image", url)
                continue
            print("Loaded")
            img = preprocess(img)
            # cv2.imwrite(f"Img {len(img_list)}.jpg", img)
            img_list.append(img)

        if len(img_list) == 0:
            return {
                "success": False,
                "message": "Cannot load image"
            }

        # remove old data
        target_label = int(class_dict[item_id])

        x_train = x_train[y_train != target_label]
        y_train = y_train[y_train != target_label]
        print('Target label', target_label)
        print(y_train)

        class_dict = {x: y for x, y in class_dict.items() if x != item_id}
        class_dict_reversed = {x: y for x, y in class_dict_reversed.items() if y != item_id}


        x = extract_features(img_list)
        y = [target_label] * len(x)
        y = np.array(y)

        x_train = np.concatenate([x_train, x])
        y_train = np.concatenate([y_train, y])
        print(y_train)

        # train the model
        clf_temp = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        clf_temp.fit(x_train, y_train)

        clf = clf_temp

        class_dict[item_id] = target_label
        class_dict_reversed[str(target_label)] = item_id

        update_storage()

        return {
            "success": True
        }
    except:
        pass

    return {
        "success": False
    }
