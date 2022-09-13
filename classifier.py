from google.cloud import storage
import io
import pickle
import os
import json
import numpy as np
import urllib.request
import random
import onnxruntime as rt
import cv2
from sklearn.neighbors import KNeighborsClassifier
import logging


logging.basicConfig(level=logging.INFO)
TARGET_SIZE = (300, 300)


def upload_cloud_storage(data_path, file_io, storage_dir):
    logging.info(f"Uploading {data_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket("obs-rmit.appspot.com")
    blob = bucket.blob(storage_dir + data_path)
    file_io.seek(0)
    blob.upload_from_file(file_io)
    logging.info(f"Uploaded {data_path}")


# async def upload_storage(data_path_list, storage, storage_dir):
#     for data_path in data_path_list[1:]:
#         if os.path.isfile(data_path):
#             logging.info(f"Uploading {data_path}")
#             storage.child(storage_dir + data_path).put("./" + data_path)
#             logging.info(f"Uploaded {data_path}")


def preprocess(img):
    img = cv2.resize(img, TARGET_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


class Classifier:
    def __init__(self, model_path, clf_path, class_path, storage_dir):
        self.model_path = model_path
        self.clf_path = clf_path
        self.class_path = class_path
        self.storage_dir = storage_dir
        self.data_path_list = [model_path, clf_path, class_path]

        # connect to firebase
        # config = {
        #     "apiKey": "AIzaSyDHDyFoXS-E7sZNJQ9G9Px9AEMStUtyrjw",
        #     "authDomain": "obs-rmit.firebaseapp.com",
        #     "databaseURL": "",
        #     "projectId": "obs-rmit",
        #     "storageBucket": "obs-rmit.appspot.com",
        #     "serviceAccount": "./obs-rmit-firebase-adminsdk.json"
        # }

        # firebase_app = pyrebase.initialize_app(config)
        # self.storage = firebase_app.storage()

        # load model and config
        self.model = None
        self.clf = None
        self.class_dict = {}
        self.class_dict_reversed = {}
        self.out_domain_threshold = 0.75

        self.load_data()

    def load_data(self):
        for data_path in self.data_path_list:
            if not os.path.isfile(data_path):
                logging.info(f"Downloading {data_path}")
                storage_client = storage.Client()
                bucket = storage_client.bucket("obs-rmit.appspot.com")
                blob = bucket.blob(self.storage_dir + data_path)
                blob.download_to_filename(data_path)
                # self.storage.child(self.storage_dir + data_path).download(self.storage_dir + data_path, "./" + data_path)
                logging.info(f"Downloaded {data_path}")

        providers = ['CPUExecutionProvider']
        self.model = rt.InferenceSession(self.model_path, providers=providers)

        # load model and config
        try:
            with open(self.clf_path, "rb") as f:
                self.clf = pickle.load(f)
            logging.info("Loaded clf")

            with open(self.class_path, "r") as f:
                classes = json.load(f)
                self.class_dict = classes["class_dict"]
                self.class_dict_reversed = classes["class_dict_reversed"]
            logging.info(f"Loaded classes {self.class_dict}")
        except:
            logging.info("File data not found")

    def get_feat(self, img):
        return self.model.run(["top_dropout"],
                              {"input_4": np.expand_dims(img, axis=0).astype("float32")})[0][0]

    def extract_features(self, img_list):
        data = []
        for i, raw_img in enumerate(img_list):
            logging.info(f"Process Image {i}")
            try:
                imgAugList = augment(raw_img)
                feats = self.model.run(["top_dropout"],
                                       {"input_4": np.array(imgAugList).astype("float32")})[0]
                data.extend(feats)
            except:
                continue
        return np.array(data)


    def get_pred(self, img):
        if self.clf is None:
            return []

        if len(self.class_dict) == 0:
            return []

        img = preprocess(img)
        feat = self.get_feat(img)
        dis, pred = self.clf.kneighbors(np.expand_dims(feat, axis=0), n_neighbors=10)
        dis = dis[0]
        pred = pred[0]
        filter_pred = []
        pred = pred[dis < self.out_domain_threshold]
        logging.info(f"{dis}")
        logging.info(f"{pred}")
        y_train = self.clf._y
        for index in pred:
            try:
                if y_train[index] not in filter_pred:
                    filter_pred.append(y_train[index])
            except:
                logging.info("Error when finding index")
                pass

        result = []
        for index in filter_pred:
            ans = self.class_dict_reversed.get(str(index), None)
            if ans is not None:
                result.append(ans)
        return result

    def update_storage(self):
        # save models
        with open(self.clf_path, "wb") as f:
            pickle.dump(self.clf, f)

        logging.info("Saved clf")

        clf_io = io.BytesIO()
        pickle.dump(self.clf, clf_io)
        upload_cloud_storage(self.clf_path, clf_io, self.storage_dir)

        class_data = {
                "class_dict": self.class_dict,
                "class_dict_reversed": self.class_dict_reversed,
            }
        with open(self.class_path, "w") as f:
            json.dump(class_data, f)
            logging.info("Saved classes")

        class_io = io.BytesIO()
        class_io.write(json.dumps(class_data).encode())
        upload_cloud_storage(self.class_path, class_io, self.storage_dir)

    def add_item(self, item_id, imageUrls):
        x_train = self.clf._fit_X
        y_train = self.clf._y
        logging.info(f"Old classes {self.class_dict}")
        if item_id in self.class_dict:
            logging.info(f"Item with id {item_id} already exists. Updating it.")
            return self.update_item(item_id, imageUrls)

        try:
            img_list = []
            for url in imageUrls:
                logging.info(f"{url}")
                try:
                    img = load_img_from_url(url)
                except:
                    logging.info(f"Cannot load image {url}")
                    continue
                logging.info("Loaded")
                img = preprocess(img)
                img_list.append(img)

            if len(img_list) == 0:
                return {
                    "success": False,
                    "message": "Cannot load image"
                }

            x = self.extract_features(img_list)
            new_label = int(np.max(y_train) + 1)
            y = [new_label] * len(x)
            y = np.array(y)

            x_train = np.concatenate([x_train, x])
            y_train = np.concatenate([y_train, y])
            logging.info(f"Labels {np.unique(y_train, return_counts=True)}")

            # train the model
            self.clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
            self.clf.fit(x_train, y_train)

            self.class_dict[item_id] = new_label
            self.class_dict_reversed[str(new_label)] = item_id
            logging.info(f"New classes {self.class_dict}")

            self.update_storage()

            return {
                "success": True,
                "message": "Added item successfully"
            }
        except:
            pass

        return {
            "success": False,
            "message": "Cannot add item"
        }

    def delete_item(self, item_id):
        x_train = self.clf._fit_X
        y_train = self.clf._y
        logging.info(f"Old classes {self.class_dict}")

        if item_id not in self.class_dict:
            return {
                "success": False,
                "message": "Item does not exist"
            }

        target_label = int(self.class_dict[item_id])

        x_train = x_train[y_train != target_label]
        y_train = y_train[y_train != target_label]
        logging.info(f"Target label {target_label}")
        logging.info(f"Labels {np.unique(y_train, return_counts=True)}")

        # train the model
        self.clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
        self.clf.fit(x_train, y_train)

        self.class_dict = {x: y for x, y in self.class_dict.items()
                           if x != item_id}
        self.class_dict_reversed = {x: y for x, y in self.class_dict_reversed.items() if y != item_id}
        logging.info(f"New classes {self.class_dict}")

        # save models
        self.update_storage()

        return {
            "success": True,
            "message": "Deleted item successfully."
        }

    def update_item(self, item_id, imageUrls):
        global clf, class_dict, class_dict_reversed
        x_train = self.clf._fit_X
        y_train = self.clf._y
        logging.info(f"Old classes {self.class_dict}")

        if item_id not in self.class_dict:
            return {
                "success": False,
                "message": "Item does not exist"
            }

        # retrain the models
        try:
            img_list = []
            for url in imageUrls:
                logging.info(f"{url}")
                try:
                    img = load_img_from_url(url)
                except:
                    logging.info(f"Cannot load image {url}")
                    continue
                logging.info("Loaded")
                img = preprocess(img)
                img_list.append(img)

            if len(img_list) == 0:
                return {
                    "success": False,
                    "message": "Cannot load image"
                }

            # remove old data
            target_label = int(self.class_dict[item_id])

            x_train = x_train[y_train != target_label]
            y_train = y_train[y_train != target_label]
            logging.info(f"Target label {target_label}")
            logging.info(f"Labels {np.unique(y_train, return_counts=True)}")


            x = self.extract_features(img_list)
            y = [target_label] * len(x)
            y = np.array(y)

            x_train = np.concatenate([x_train, x])
            y_train = np.concatenate([y_train, y])
            logging.info(f"Labels {np.unique(y_train, return_counts=True)}")

            # train the model
            self.clf = KNeighborsClassifier(n_neighbors=3, metric="cosine")
            self.clf.fit(x_train, y_train)

            logging.info(f"New classes {self.class_dict}")

            self.update_storage()

            return {
                "success": True,
                "message": "Update item successfully"
            }
        except:
            pass

        return {
            "success": False,
            "message": "Cannot update item"
        }
