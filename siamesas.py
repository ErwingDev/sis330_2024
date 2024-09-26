import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Módulo de atención para enfocar partes relevantes
class Attention(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, num_embeddings, embedding_dim)
        weights = self.attention(x)  # Calcular pesos de atención
        weights = torch.softmax(weights, dim=1)  # Normalizar los pesos
        return torch.sum(weights * x, dim=1)  # Ponderar los embeddings

# Definir la red siamesa con embeddings multimodales
class SiameseNetworkWithAttention(nn.Module):
    def __init__(self):
        super(SiameseNetworkWithAttention, self).__init__()
        
        # Modelo preentrenado para rostro, cuerpo, y ropa
        self.resnet_face = models.resnet18(pretrained=True)
        self.resnet_body = models.resnet18(pretrained=True)
        self.resnet_clothes = models.resnet18(pretrained=True)

        # Quitar la última capa de clasificación
        self.resnet_face = nn.Sequential(*list(self.resnet_face.children())[:-1])
        self.resnet_body = nn.Sequential(*list(self.resnet_body.children())[:-1])
        self.resnet_clothes = nn.Sequential(*list(self.resnet_clothes.children())[:-1])

        # Atención para combinar los embeddings
        self.attention = Attention(in_features=512, hidden_size=128)
    
    def forward(self, x_face, x_body, x_clothes):
        # Extraer características de las diferentes ramas
        embedding_face = self.resnet_face(x_face).view(x_face.size(0), -1)
        embedding_body = self.resnet_body(x_body).view(x_body.size(0), -1)
        embedding_clothes = self.resnet_clothes(x_clothes).view(x_clothes.size(0), -1)

        # Concatenar embeddings para formar un solo vector multimodal
        combined_embeddings = torch.stack([embedding_face, embedding_body, embedding_clothes], dim=1)
        
        # Aplicar mecanismo de atención para obtener un embedding final ponderado
        attention_embedding = self.attention(combined_embeddings)
        return attention_embedding

# Transformaciones de las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Función para calcular la distancia euclidiana entre embeddings
def euclidean_distance(x1, x2):
    return F.pairwise_distance(x1, x2)

# Cargar la imagen de referencia y obtener su embedding multimodal
def get_multimodal_embedding(model, image, trim = False):
    # image = Image.open(image_path).convert('RGB')
    if trim :
        w, h = image.size
        image40 = image.crop((0, 0, w, int((h*40)/100)))
        image70 = image.crop((0, 0, w, int((h*70)/100)))
        image_tensor1 = transform(image).unsqueeze(0)
        image_tensor2 = transform(image40).unsqueeze(0)
        image_tensor3 = transform(image70).unsqueeze(0)
    else :
        image_tensor1 = transform(image).unsqueeze(0)
        image_tensor2 = transform(image).unsqueeze(0)
        image_tensor3 = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor1, image_tensor2, image_tensor3)  # Para rostro, cuerpo y ropa
    return embedding



""" model = SiameseNetworkWithAttention()
model.eval() """

""" model_person = YOLO('./utils/model_people3.pt')
model_fashion = YOLO('./utils/model_fashion.pt')

reference_image_path = 'video/busqueda_inversa/prueba3.jpg' """

""" reference_embedding = get_multimodal_embedding(model, Image.open(reference_image_path).convert('RGB'), True)
embedding_original = reference_embedding """

""" class_clothes = ['Tshirt', 'dress', 'jacket', 'pants', 'shirt', 'short', 'skirt', 'sweater']
clothes_selected = [] """


def getInfoClothes(imgOri, results, class_clothes) :
    rspta = [] 
    img_cv = np.array(imgOri)
    for result in results:
        boxes = result.boxes
        for box in boxes :
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = class_clothes[int(box.cls[0])]
            cropped_img = img_cv[y1:y2, x1:x2]
            hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
            predominant_color = np.argmax(hist)
            rspta.append({
                'name': label, 
                'color': predominant_color
            })
    return rspta


def getcolor_fashion(name, clothes_selected) :
    color = None
    for fashion in clothes_selected:
        if fashion['name'].lower() == name.lower() and name in {'Tshirt', 'dress', 'jacket', 'shirt', 'sweater'} :
            color = fashion['color']
            break
    return color


def exists_color(color, clothes_selected) :
    exists = False
    for fashion in clothes_selected :                                                                        # 'camiseta', 'vestido', 'chaqueta', 'camisa', 'suéter'
        if int(fashion['color'])-2 <= color and color <= int(fashion['color'])+2 and fashion['name'].lower() in {'tshirt', 'dress', 'jacket', 'shirt', 'sweater'} :
            exists = True
    return exists


