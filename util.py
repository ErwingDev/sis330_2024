
import os
import random
import torch
import torchreid
from deep_sort_realtime.deepsort_tracker import DeepSort
import base64
from PIL import Image
from io import BytesIO

class Util :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = DeepSort(max_age=7, nn_budget=100)
    extractor = torchreid.utils.FeatureExtractor(
        model_name = 'osnet_x1_0',
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_path = './models/model.pth-osnet.tar-300',
    )

    def extract_features(self, person_image):
        features = self.extractor([person_image])
        return features
    
    def compare_embeddings(self, features_1, features_2):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cos(features_1, features_2)
        return similarity
    
    def get_tracks(self, results, frame) :
        aux = []
        for result in results:
            boxes = result.boxes
            for box in boxes :
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                width, height = x_max - x_min, y_max - y_min
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                aux.append(([x_min, y_min, width, height], confidence, class_id))

        tracks = self.tracker.update_tracks(aux, frame = frame)
        list_ids = [int(track.track_id) for track in tracks if track.is_confirmed()]

        return {
            'tracks': tracks,
            'ids': list_ids
        }
    def save_images_base64(self, faces, folder_name) :
        index = 1
        for base64_string in faces :
            if "data:image" in base64_string:
                base64_string = base64_string.split(",")[1]
            image_bytes = base64.b64decode(base64_string)
            image_stream = BytesIO(image_bytes)
            image = Image.open(image_stream)
            image.save(folder_name+"/image"+str(index)+".jpg")
            index = index + 1

    def save_image_one_base64(self, file, folder_name, file_name) :
        if "data:image" in file:
            file = file.split(",")[1]
        image_bytes = base64.b64decode(file)
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        image.save(folder_name+"/"+file_name+".jpg")

    def get_image_of_base64(self, file) : 
        if "data:image" in file:
            file = file.split(",")[1]
        image_bytes = base64.b64decode(file)
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        return image
    
    def get_token(self, size = 5) : 
        return ''.join(str(random.randint(0, 9)) for _ in range(size))

    def get_index_file(self, carpeta) :
        index = '0001'
        files = [archivo for archivo in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, archivo))]
        if len(files) > 0 :
            last_file = files[len(files)-1]
            num_person = int(last_file.split('_')[0]) + 1
            index = f"{num_person:04d}"
        return index


