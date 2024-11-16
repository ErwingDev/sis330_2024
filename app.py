from flask import Flask, render_template, request, jsonify, abort  # Importa las clases y funciones necesarias de Flask.
from deep_sort.deep_sort import DeepSort
from models1 import db, Event, Participant, Attendance, participants_events  # Importa los modelos y la conexión a la base de datos.
import threading  # Importa 'threading' para manejar la ejecución en hilos separados.
import cv2  # Importa OpenCV para procesar y analizar imágenes.
import time  # Importa 'time' para manejar funciones relacionadas con el tiempo.
import torch  # Importa PyTorch para procesamiento y entrenamiento de modelos de reconocimiento facial.
import yaml  # Importa 'yaml' para manejar archivos YAML, generalmente usados para configuraciones.
from recognize import recognition, mapping_bbox, norm_crop  # Importa funciones de reconocimiento facial y procesamiento de imágenes.
from face_detection.scrfd.detector import SCRFD  # Importa el detector de rostros SCRFD.
from face_detection.yolov5_face.detector import Yolov5Face  # Importa el detector de rostros basado en Yolov5.
from face_recognition.arcface.model import iresnet_inference  # Importa la función de inferencia del modelo ResNet para reconocimiento facial.
from face_recognition.arcface.utils import compare_encodings, read_features  # Importa funciones para comparar codificaciones faciales y leer características.
from face_tracking.tracker.byte_tracker import BYTETracker  # Importa BYTETracker para seguimiento de rostros en video.
from face_tracking.tracker.visualize import plot_tracking  # Importa 'plot_tracking' para visualizar el rastreo en video.
from flask_socketio import SocketIO, emit  # Importa 'SocketIO' y 'emit' para manejar comunicación en tiempo real con WebSockets.
import os
import base64
from PIL import Image
from io import BytesIO
import subprocess
from ultralytics import YOLO
import siamesas
import numpy as np
import torchreid
import torch
from torchvision import transforms
from scipy.spatial.distance import cosine

#from threading import Event

model_person = YOLO('./models/model_people.pt')
model_fashion = YOLO('./models/model_fashion.pt')

#app = Flask(__name__)
app = Flask(__name__, static_folder='static')  # Crea una instancia de la aplicación Flask y define la carpeta 'static' para archivos estáticos.
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/face_recognition_db'  # Configura la URI de la base de datos MySQL para SQLAlchemy.
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Desactiva el seguimiento de modificaciones para ahorrar recursos.
db.init_app(app)  # Inicializa la aplicación Flask con la base de datos utilizando SQLAlchemy.



# Inicializa SocketIO con la aplicación Flask
socketio = SocketIO(app)  # Inicializa SocketIO para habilitar comunicación en tiempo real en la aplicación Flask.
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Configura el dispositivo para PyTorch, usando GPU si está disponible, de lo contrario CPU.
#stop_tracking_event = Event()  # Comentado: Podría haber sido usado para detener el seguimiento de rostros, pero está desactivado.

# Load models and features
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_10g_bnkps.onnx")  # Carga el modelo SCRFD para detección de rostros desde el archivo ONNX.
#detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")  # Comentado: Alternativa para cargar el modelo Yolov5 para detección de rostros.
recognizer = iresnet_inference(model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device)  # Carga el modelo ArcFace para reconocimiento facial usando ResNet.
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")  # Lee las características faciales prealmacenadas y sus nombres desde la ruta especificada.


model_reid = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=3,
    loss="softmax",
    pretrained=True  
)
model_reid.to(device)
model_reid.eval()
torchreid.utils.load_pretrained_weights(model_reid, './models/model.pth.tar-300')
transform = transforms.Compose([
    transforms.ToTensor(),
])

def extract_features(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model_reid(image)
    return features.cpu().numpy()[0]

def compare_embeddings(features_1, features_2):
    similarity = 1 - cosine(features_1, features_2)
    return similarity

def list_cameras():  # Define una función para listar las cámaras disponibles.
    index = 0  # Inicializa el índice de la cámara.
    arr = []  # Crea una lista vacía para almacenar los índices de las cámaras detectadas.
    i = 10  # Limitar la búsqueda a 10 índices de cámaras.
    while i > 0:  # Itera mientras i sea mayor que 0.
        cap = cv2.VideoCapture(index)  # Intenta abrir la cámara en el índice actual.
        if cap.read()[0]:  # Si la cámara está disponible y se puede leer un frame, se confirma que existe.
            arr.append(index)  # Añade el índice de la cámara a la lista.
        cap.release()  # Libera la cámara para que otros procesos puedan usarla.
        index += 1  # Incrementa el índice para probar la siguiente cámara.
        i -= 1  # Decrementa el contador i para limitar la búsqueda a 10 cámaras.
    return arr  # Devuelve la lista de índices de cámaras detectadas.

def load_config(file_name):  # Define una función para cargar configuraciones desde un archivo YAML.
    with open(file_name, "r") as stream:  # Abre el archivo YAML en modo lectura.
        try:
            return yaml.safe_load(stream)  # Intenta cargar y devolver el contenido del archivo YAML.
        except yaml.YAMLError as exc:  # Captura cualquier error de YAML.
            print(exc)  # Imprime el error si ocurre.

# Shared data structure

recognized_participants = []  # Crea una lista para almacenar los participantes reconocidos.
tracking_threads = []  # Crea una lista para almacenar los hilos que manejarán el seguimiento de rostros.
camera_instances = {}  # Crea un diccionario para almacenar las instancias de las cámaras activas.
stop_tracking_event = threading.Event()  # Crea un evento de hilo para señalizar cuando detener el seguimiento de rostros.
stop_tracking_detect_person = threading.Event()  # Crea un evento de hilo para señalizar cuando detener el seguimiento de rostros.
stop_tracking_find_person = threading.Event()  # Crea un evento de hilo para señalizar cuando detener el seguimiento de rostros.


@app.route('/')  # Define la ruta para la página principal de la aplicación.
def index():  # Define la función que se ejecutará cuando se acceda a la ruta principal.
    return render_template('add-face.html')  # Renderiza la plantilla 'index.html' pasando los eventos, cámaras y participantes reconocidos como contexto.

import logging  # Importa el módulo logging para registrar mensajes de depuración.

logging.basicConfig(level=logging.DEBUG)  # Configura el nivel de logging a DEBUG, lo que permite registrar mensajes detallados para depuración.

def saveImagesBase64(faces, folder_name) :
    index = 1
    for base64_string in faces :
        if "data:image" in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        image.save(folder_name+"/image"+str(index)+".jpg")
        index = index + 1

def saveImageOneBase64(file, folder_name, file_name) :
    if "data:image" in file:
        file = file.split(",")[1]
    image_bytes = base64.b64decode(file)
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream)
    image.save(folder_name+"/"+file_name+".jpg")

def getImageOfBase64(file) : 
    if "data:image" in file:
        file = file.split(",")[1]
    image_bytes = base64.b64decode(file)
    image_stream = BytesIO(image_bytes)
    image = Image.open(image_stream)
    return image


@app.route('/save-faces', methods=['POST'])
def save_faces(): 
    name = str.lower(request.form['name']) 
    ci = request.form['ci']
    faces = request.form.getlist('photo[]')

    folder_name = 'datasets/new_persons/'+ci+'__'+name.replace(" ", "_")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        # print(f"Carpeta '{folder_name}' creada.")
    
    saveImagesBase64(faces, folder_name)
    try:
        result = subprocess.run(['python', 'add_persons.py'], capture_output=True, text=True)
        print("Salida del script add.py:")
        print(result.stdout)
        
        if result.stderr:
            print("Error en el script:")
            print(result.stderr)

    except Exception as e:
        print(f"Ocurrió un error al ejecutar el script: {e}")
        return jsonify({
            'status': 400,
            'message': 'Ocurrió un error.'
        })

    return jsonify({
        'status': 200,
        'message': 'Guardado correctamente.'
    })

@app.route('/add-person')
def render_add_person(): 
    """ cameras = list_cameras()
    print("Cámaras disponibles:", cameras) """
    cameras = list_cameras()
    # print("Cámaras seleccionadas:", request.args.getlist('camera'))
    # return render_template('add-person.html', cameras=cameras)
    return render_template('add-person.html', cameras=cameras)


@app.route('/save-crop', methods=['POST'])
def save_crop(): 
    person = str.lower(request.form['person'])
    img = request.form['image']
    """ name_person = person.split('__')
    name_person = name_person[1] if len(name_person) > 1 else name_person  """                               
    saveImageOneBase64(img, 'files', person)
    return jsonify({
        'status': 200,
        'message': 'Guardado correctamente.'
    })

@app.route('/get_capture_body', methods=['POST'])
def get_capture_body() :
    data = data = request.json  
    image = data['image']

    frame_crop = ""
    frame = getImageOfBase64(image).convert('RGB')
    frame = np.array(frame)
    results = model_person(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes :
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            probability = ((box.conf[0]*100)/100).item()
            if probability > 0.75 :
                img_crop = frame[y1:y2, x1:x2]
                _, buffer_crop = cv2.imencode('.jpg', img_crop)
                frame_crop = base64.b64encode(buffer_crop).decode('utf-8')
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('person_recognized', {'body_person': frame_base64, 'status': '', 'person_crop': frame_crop, 'type': 'body'})
                    
    
    return jsonify({
        'status': 200,
        'message': frame_crop
    })


@app.route('/detect_body', methods=['POST']) 
def detect_body(): 
    data = request.json  
    camera_indices = data['camera_indices']

    global stop_tracking_detect_person
    stop_tracking_detect_person.clear()

    for camera_index in camera_indices:
        with app.app_context():  # Crea un contexto de aplicación.
            cap = cv2.VideoCapture(int(camera_index))  # Abre la cámara especificada.
            if not cap.isOpened():  # Verifica si la cámara se abrió correctamente.
                print(f"Error al abrir la cámara {camera_index}")
                return
            print(f"Stop.tracking:  WHIT {stop_tracking_event.is_set()}") 
            try:
                while not stop_tracking_detect_person.is_set():
                    ret_val, frame = cap.read()  # Lee un cuadro de la cámara.
                        #time.sleep(1)  # Espera un poco antes de leer el siguiente cuadro.
                    if not ret_val:  # Verifica si se pudo leer un cuadro.
                        print("Error al leer desde la cámara. Reintentando...")
                        continue

                    results = model_person(frame)

                    for result in results:
                        boxes = result.boxes
                        for box in boxes :
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            probability = ((box.conf[0]*100)/100).item()
                            if probability > 0.75 :
                                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                """ _, frame_person = cap.read()
                                frame_rgb = cv2.cvtColor(frame_person, cv2.COLOR_BGR2RGB)
                                img_person = frame_rgb[y1:y2, x1:x2] """
                                h = y2 - y1
                                frame_h, frame_w, canales = frame.shape                                
                                h_ratio = h / frame_h
                                # cv2.putText(frame, str(h_ratio), (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                # if h_ratio < 0.75 : 
                                frame_crop = ''
                                if h_ratio <= 0.96 : 
                                # if h_ratio <= 0.85 : 
                                    print(frame)
                                    # cv2.putText(frame, str(h_ratio), (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    img_crop = frame[y1:y2, x1:x2]
                                    _, buffer_crop = cv2.imencode('.jpg', img_crop)
                                    frame_crop = base64.b64encode(buffer_crop).decode('utf-8')
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                _, buffer = cv2.imencode('.jpg', frame)
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                socketio.emit('person_recognized', {'body_person': frame_base64, 'status': h_ratio, 'person_crop': frame_crop, 'type': 'body'})
                                
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite cerrar la ventana presionando 'q'.
                        break

                    time.sleep(0.01)  # Controla la velocidad del bucle.
            finally:
                cap.release()  # Libera la cámara al finalizar.
                cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV.
    return jsonify({'status': 'Detect Body started'})


@app.route('/stop_detect_body', methods=['POST'])
def stop_detect_body(): 
    global stop_tracking_detect_person
    stop_tracking_detect_person.set() 

    for camera_index, cap in camera_instances.items():  
        cap.release() 

    camera_instances.clear()
    return jsonify({'status': 'Detect Boyd stopped'})

@app.route('/find-person')
def render_find_person() : 
    cameras = list_cameras()
    return render_template('find-person.html', cameras=cameras)

@app.route('/get-list-persons', methods=['POST'])
def get_list_persons(): 
    # path_folder = "files/"
    path_folder = "datasets/data/"
    list = [f for f in os.listdir(path_folder)]
    return jsonify({'data': list})

@app.route('/stop-find-person', methods=['POST'])
def stop_find_person(): 
    global stop_tracking_find_person
    stop_tracking_find_person.set() 

    for camera_index, cap in camera_instances.items():  
        cap.release() 

    camera_instances.clear()
    return jsonify({'status': 'Find Person stopped'})


@app.route('/start-find-person', methods=['POST'])
def start_find_person():
    file = request.form['file']
    name = request.form['name']
    ci = request.form['ci']
    camera_index = request.form['camera']
    find_name = file.split('__')[1]

    global stop_tracking_find_person
    stop_tracking_find_person.clear()

    deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
    trackerDeep = DeepSort(model_path=deep_sort_weights, max_age=15)
    config_tracking = load_config("./face_tracking/config/config_tracking.yaml")
    tracker = BYTETracker(args=config_tracking, frame_rate=30)
    idtrack = None
    matched_name = ""
    find_person = False
    prob_seg = 0
    person_seg = None

    reference_image_path = 'files/'+file+'.jpg'
    target_features = extract_features(Image.open(reference_image_path).convert('RGB'))

    with app.app_context():  # Crea un contexto de aplicación.
        cap = cv2.VideoCapture(int(camera_index))  # Abre la cámara especificada.
        if not cap.isOpened():  # Verifica si la cámara se abrió correctamente.
            print(f"Error al abrir la cámara {camera_index}")
            return
        try:
            while not stop_tracking_find_person.is_set():
                ret_val, frame = cap.read()  # Lee un cuadro de la cámara.
                if not ret_val:  # Verifica si se pudo leer un cuadro.
                    print("Error al leer desde la cámara. Reintentando...")
                    continue

                # '''
                results = model_person.predict(source=frame, stream=True, conf=0.70, iou=0.5)
                for result in results:
                    boxes = result.boxes  # Boxes object for bbox outputs
                    probs = result.probs  # Class probabilities for classification outputs
                    cls = boxes.cls.tolist()  # Convert tensor to list
                    xyxy = boxes.xyxy
                    conf = boxes.conf
                    xywh = boxes.xywh


                conf = conf.detach().cpu().numpy()
                xyxy = xyxy.detach().cpu().numpy()
                bboxes_xywh = xywh
                bboxes_xywh = xywh.cpu().numpy()
                bboxes_xywh = np.array(bboxes_xywh, dtype=float)
                tracks = trackerDeep.update(bboxes_xywh, conf, frame)
                list_ids = [int(track.track_id) for track in trackerDeep.tracker.tracks if track.is_confirmed()]
                for track in trackerDeep.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 0: 
                        continue

                    track_id = track.track_id
                    hits = track.hits
                    x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
                    w = x2 - x1  # Calculate width
                    h = y2 - y1  # Calculate height

                    # Set color values for red, blue, and green
                    color = (0, 255, 0)  # (B, G, R)

                    person_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    person_features = extract_features(person_image)
                    similarity = compare_embeddings(person_features, target_features)

                    threshold = 0.7
                    if similarity > threshold :
                        if similarity.item() > prob_seg : 
                            prob_seg = similarity.item()
                            idtrack = track_id

                    if track_id == idtrack :
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                    if idtrack and int(idtrack) not in list_ids :
                        idtrack = None
                        prob_seg = 0
                        continue
                    # '''

                '''
                results = model_person.predict(source=frame, stream=True, conf=0.70, iou=0.5)
                for result in results:
                    boxes = result.boxes
                    for box in boxes :
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        person = frame[y1:y2, x1:x2]
                        person_image = person
                        person_features = extract_features(person_image)
                        similarity = compare_embeddings(person_features, target_features)
                        # threshold = 0.65
                        threshold = 0.7

                        if similarity > threshold:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    '''
                            
                
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('person_recognized', {'frame_find_person': frame_base64})
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite cerrar la ventana presionando 'q'.
                    break

                time.sleep(0.01) 
        finally:
            cap.release()  
            cv2.destroyAllWindows() 

    return jsonify({'status': 'Find Person started'})


@app.route('/start_tracking', methods=['POST'])  # Define la ruta para iniciar el seguimiento de rostros, utilizando el método POST.
def start_tracking():  # Define la función que se ejecutará cuando se acceda a la ruta '/start_tracking'.
    data = request.json  # Obtiene los datos JSON enviados en la solicitud.
    camera_indices = data['camera_indices']  # Lista de índices de cámaras para iniciar el seguimiento.
    print(camera_indices)
    #TODO: jalar las variables y los metodos iniciantes para el tracking
    config_tracking = load_config("./face_tracking/config/config_tracking.yaml")  # Carga la configuración de seguimiento desde un archivo YAML.

    stop_tracking_event.clear()  # Limpia el evento de parada para asegurar que el seguimiento continúe.

    for camera_index in camera_indices:  # Recorre cada índice de cámara seleccionado.
        logging.debug(f"Starting tracker for camera index: {camera_index}")  # Registra un mensaje de depuración indicando qué cámara se está iniciando.
        tracker = BYTETracker(args=config_tracking, frame_rate=30)  # Crea una instancia del rastreador BYTETracker con la configuración y una tasa de cuadros de 30 fps.
        cap = cv2.VideoCapture(int(camera_index))  # Abre la cámara especificada por el índice.
        camera_instances[camera_index] = cap  # Almacena la instancia de la cámara en el diccionario.
        thread = threading.Thread(target=process_tracking, args=(camera_index, detector, tracker))  # Crea un nuevo hilo para ejecutar el seguimiento en paralelo.
        #SOSPECHOSO
        thread.start()  # Inicia el hilo.
        tracking_threads.append(thread)  # Almacena el hilo en la lista de hilos de seguimiento.

    return jsonify({'status': 'Tracking started'})  # Devuelve una respuesta JSON indicando que el seguimiento ha comenzado.

@app.route('/stop_tracking', methods=['POST'])  # Define la ruta para detener el seguimiento de rostros, utilizando el método POST.
def stop_tracking():  # Define la función que se ejecutará cuando se acceda a la ruta '/stop_tracking'.

    global tracking_threads, stop_tracking_event, camera_instances  # Declara variables globales para manejar los hilos de seguimiento, el evento de parada y las instancias de cámaras.
    print(f"PARANDO TRACKING XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX") 
    stop_tracking_event.set()  # Señala a los hilos que deben detenerse activando el evento de parada.

    for thread in tracking_threads:  # Espera a que todos los hilos de seguimiento terminen.
        thread.join()  # Bloquea la ejecución hasta que cada hilo termine.

    for camera_index, cap in camera_instances.items():  # Recorre cada instancia de cámara almacenada.
        cap.release()  # Libera la cámara correspondiente.

    tracking_threads = []  # Limpia la lista de hilos de seguimiento.
    camera_instances.clear()  # Limpia el diccionario de instancias de cámaras.

    return jsonify({'status': 'Tracking stopped, participants list cleared, and cameras released'})  # Devuelve una respuesta JSON indicando que el seguimiento se ha detenido y las cámaras han sido liberadas.

def process_tracking(camera_index, detector, tracker):  # Define la función para procesar el seguimiento de una cámara.
    global stop_tracking_event, recognized_participants  # Declara variables globales.

    with app.app_context():  # Crea un contexto de aplicación.
        cap = cv2.VideoCapture(int(camera_index))  # Abre la cámara especificada.
        if not cap.isOpened():  # Verifica si la cámara se abrió correctamente.
            print(f"Error al abrir la cámara {camera_index}")
            return
        print(f"Stop.tracking:  WHIT {stop_tracking_event.is_set()}") 
        try:
            print(f"Stop.tracking event.is_set() WHILE pre: {stop_tracking_event.is_set()}")  # Imprime el estado del evento de parada
            while not stop_tracking_event.is_set():  # Continúa el bucle mientras el evento de parada no esté establecido.
                print(f"Stop.tracking: WHILE {stop_tracking_event.is_set()}")
                ret_val, frame = cap.read()  # Lee un cuadro de la cámara.
                #time.sleep(1)  # Espera un poco antes de leer el siguiente cuadro.
                if not ret_val:  # Verifica si se pudo leer un cuadro.
                    print("Error al leer desde la cámara. Reintentando...")
                    continue

                # Lógica de detección y seguimiento...

                # Detección de caras y seguimiento
                outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
                if outputs is not None:
                    online_targets = tracker.update(outputs, [img_info["height"], img_info["width"]], (128, 128))
                    tracking_bboxes = []
                    tracking_ids = []

                    for t in online_targets:
                        print(f"Stop.tracking: FOR t {stop_tracking_event.is_set()}")
                        tlwh = t.tlwh  # bounding box en formato [x, y, width, height]
                        tracking_id = t.track_id
                        x, y, w, h = map(int, tlwh)
                        tracking_bboxes.append([x, y, x + w, y + h])
                        tracking_ids.append(tracking_id)

                    for tid, tbbox in zip(tracking_ids, tracking_bboxes):
                        print(f"Stop.tracking: FOR tid {stop_tracking_event.is_set()}")
                        x_min, y_min, x_max, y_max = map(int, tbbox)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        matched_name = "Unknown"
                        highest_score = 0.0

                        for j, dbbox in enumerate(bboxes):
                            print(f"Stop.tracking: FOR j {stop_tracking_event.is_set()}") 

                            similarity_score = mapping_bbox(tbbox, dbbox)
                            print(f"Similitud entre tbbox y dbbox: {similarity_score}")
                            if similarity_score > 0.9:
                                img = img_info["raw_img"]
                                face_alignment = norm_crop(img, landmark=landmarks[j])

                                if face_alignment is not None:
                                    print(f"Procesando cara alineada: Dimensiones - {face_alignment.shape}")
                                else:
                                    print("Error: face_alignment es None")
                                    continue

                                print("Antes de recognition: ")
                                print(f"Imagen alineada tipo: {type(face_alignment)}, dimensiones: {face_alignment.shape}")

                                score, name = recognition(face_alignment)

                                print(f"Resultado de recognition -> Score: {score}, Nombre: {name}")

                                if score > highest_score:
                                    highest_score = score
                                    matched_name = name if score > 0.25 else "Unknown"

                                # if score > 0.75 :
                                if score > 0.60 :
                                    name_person = name.split('__')
                                    name_person = name_person[1] if len(name_person) > 1 else name_person
                                    cv2.putText(frame, f'{name_person}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    # _, buffer = cv2.imencode('.jpg', frame)
                                    # frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                    # socketio.emit('person_recognized', {'person': name, 'frame': frame_base64})
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                socketio.emit('person_recognized', {'person': name, 'frame': frame_base64, 'type': 'face'})

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite cerrar la ventana presionando 'q'.
                    break

                time.sleep(0.01)  # Controla la velocidad del bucle.
        finally:
            cap.release()  # Libera la cámara al finalizar.
            cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV.


if __name__ == "__main__":  # Comprueba si el script se está ejecutando directamente.
    
    with app.app_context():  # Crea un contexto de aplicación para ejecutar el siguiente bloque de código.
        db.create_all()  # Crea todas las tablas en la base de datos según los modelos definidos.

    socketio.run(app, debug=False)  # Inicia la aplicación Flask con SocketIO, sin habilitar el modo de depuración.


