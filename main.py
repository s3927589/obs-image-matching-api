from fastapi import FastAPI, HTTPException, status, Request, Response, File , UploadFile
# from tensorflow.keras.models import load_model
import pickle
import json
import numpy as np
import urllib.request
import random
from tqdm import tqdm
import onnxruntime as rt
import cv2
from sklearn.neighbors import KNeighborsClassifier
from models import AddItemForm


app = FastAPI()
TARGET_SIZE = (300, 300)
# model = load_model("./efficient.h5")
providers = ['CPUExecutionProvider']
model = rt.InferenceSession("./efficient.onnx", providers=providers)


with open("./knn.pickle", "rb") as f:
    clf = pickle.load(f)

with open("./data.npy", "rb") as f:
    x_train = pickle.load(f)

with open("./labels.npy", "rb") as f:
    y_train = pickle.load(f)

with open("./classes.json", "r") as f:
    classes = json.load(f)
    class_dict = classes["class_dict"]
    class_dict_reversed = classes["class_dict_reversed"]

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
    _, pred = clf.kneighbors(np.expand_dims(feat, axis=0), n_neighbors=4)
    pred = pred[0]
    filter_pred = []
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
    global x_train, y_train, clf, class_dict, class_dict_reversed
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

        x_train_temp = np.concatenate([x_train, x])
        y_train_temp = np.concatenate([y_train, y])
        print(y_train_temp)

        # train the model
        clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        clf.fit(x_train_temp, y_train_temp)

        x_train = x_train_temp
        y_train = y_train_temp

        class_dict[item_id] = new_label
        class_dict_reversed[str(new_label)] = item_id

        # save models
        with open("knn.pickle", "wb") as f:
            pickle.dump(clf, f)
        with open("data.npy", "wb") as f:
            pickle.dump(x_train, f)
        with open("labels.npy", "wb") as f:
            pickle.dump(y_train, f)
        with open("classes.json", "w") as f:
            json.dump({
                "class_dict": class_dict,
                "class_dict_reversed": class_dict_reversed,
            }, f)

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
    global x_train, y_train, clf, class_dict, class_dict_reversed
    if item_id not in class_dict:
        return {
            "success": False,
            "message": "Item does not exist"
        }

    target_label = int(class_dict[item_id])

    x_train_temp = x_train[y_train != target_label]
    y_train_temp = y_train[y_train != target_label]
    print('Target label', target_label)
    print(y_train_temp)

    # train the model
    clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    clf.fit(x_train_temp, y_train_temp)

    x_train = x_train_temp
    y_train = y_train_temp

    class_dict = {x: y for x, y in class_dict.items() if x != item_id}
    class_dict_reversed = {x: y for x, y in class_dict_reversed.items() if y != item_id}

    # save models
    with open("knn.pickle", "wb") as f:
        pickle.dump(clf, f)
    with open("data.npy", "wb") as f:
        pickle.dump(x_train, f)
    with open("labels.npy", "wb") as f:
        pickle.dump(y_train, f)
    with open("classes.json", "w") as f:
        json.dump({
            "class_dict": class_dict,
            "class_dict_reversed": class_dict_reversed,
        }, f)

    return {
        "success": True,
        "message": "Deleted item successfully."
    }


@app.put('/api/update_item', status_code=status.HTTP_200_OK)
async def update_item(data: AddItemForm):
    item_id = data.item_id
    imageUrls = data.imageUrls
    global x_train, y_train, clf, class_dict, class_dict_reversed

    # delete old data
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

    class_dict = {x: y for x, y in class_dict.items() if x != item_id}
    class_dict_reversed = {x: y for x, y in class_dict_reversed.items() if y != item_id}

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

        x = extract_features(img_list)
        y = [target_label] * len(x)
        y = np.array(y)

        x_train_temp = np.concatenate([x_train, x])
        y_train_temp = np.concatenate([y_train, y])
        print(y_train_temp)

        # train the model
        clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        clf.fit(x_train_temp, y_train_temp)

        x_train = x_train_temp
        y_train = y_train_temp

        class_dict[item_id] = target_label
        class_dict_reversed[str(target_label)] = item_id

        # save models
        with open("knn.pickle", "wb") as f:
            pickle.dump(clf, f)
        with open("data.npy", "wb") as f:
            pickle.dump(x_train, f)
        with open("labels.npy", "wb") as f:
            pickle.dump(y_train, f)
        with open("classes.json", "w") as f:
            json.dump({
                "class_dict": class_dict,
                "class_dict_reversed": class_dict_reversed,
            }, f)

        return {
            "success": True
        }
    except:
        pass

    return {
        "success": False
    }
