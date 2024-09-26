from flask import Flask, render_template, request, jsonify, abort  # Importa las clases y funciones necesarias de Flask.
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
                            if probability > 0.60 :
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
                                    # cv2.putText(frame, str(h_ratio), (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    img_crop = frame[y1:y2, x1:x2]
                                    _, buffer_crop = cv2.imencode('.jpg', img_crop)
                                    frame_crop = base64.b64encode(buffer_crop).decode('utf-8')
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                _, buffer = cv2.imencode('.jpg', frame)
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                socketio.emit('person_recognized', {'body_person': frame_base64, 'status': h_ratio, 'person_crop': frame_crop})
                                
                    
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
    path_folder = "files/"
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

    global stop_tracking_find_person
    stop_tracking_find_person.clear()

    model_s = siamesas.SiameseNetworkWithAttention()
    model_s.eval()
    reference_image_path = 'files/'+file

    reference_embedding = siamesas.get_multimodal_embedding(model_s, Image.open(reference_image_path).convert('RGB'), True)
    embedding_original = reference_embedding
    #           'Camiseta', 'vestido', 'chaqueta', 'pantalones', 'camisa', 'short', 'falda', 'suéter'
    class_clothes = ['Tshirt', 'dress', 'jacket', 'pants', 'shirt', 'short', 'skirt', 'sweater']

    imgOri = Image.open(reference_image_path).convert('RGB')
    result_clothes = model_fashion(reference_image_path, stream=False)
    info_clothes = siamesas.getInfoClothes(imgOri, result_clothes, class_clothes)

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

                persons = []
                results = model_person(frame) 

                for result in results:
                    boxes = result.boxes
                    for box in boxes :
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        probability = ((box.conf[0]*100)/100).item()
                        # if probability > 0.75 :
                        if probability > 0.60 :
                            _, frame_person = cap.read()
                            frame_rgb = cv2.cvtColor(frame_person, cv2.COLOR_BGR2RGB)
                            img_person = frame_rgb[y1:y2, x1:x2]
                            persons.append({
                                'img': img_person,
                                'box': {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2},
                            })  

                if len(persons) > 0 :
                    for person in persons:
                        frame_pil = Image.fromarray(person['img'])
                        frame_tensor = siamesas.transform(frame_pil).unsqueeze(0)
                        
                        with torch.no_grad():
                            frame_embedding = model_s(frame_tensor, frame_tensor, frame_tensor)  

                        condition_fashion = False
                        aux = ""
                        results_clothes = model_fashion(person['img'])
                        for r in results_clothes :
                            boxes_ = r.boxes
                            for box_ in boxes_ :
                                x1_, y1_, x2_, y2_ = box_.xyxy[0]
                                x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
                                label = class_clothes[int(box_.cls[0])]
                                # cropped_img = img_person[y1_:y2_, x1_:x2_]
                                cropped_img = person['img'][y1_:y2_, x1_:x2_]
                                hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                                hist = cv2.calcHist([hsv_img], [0], None, [256], [0, 256])
                                predominant_color = np.argmax(hist)
                                # ori_color = None

                                condition_fashion = siamesas.exists_color(predominant_color, info_clothes)
                                print(condition_fashion)

                                # aux = label+" "+str(ori_color)
                                # print(label, predominant_color, ori_color)
                                # if ori_color and (ori_color <= predominant_color+10 and predominant_color-10 <= ori_color) :
                                # if ori_color and (ori_color <= predominant_color+5 and predominant_color-5 <= ori_color) :

                        distance = siamesas.euclidean_distance(reference_embedding, frame_embedding)
                        # cv2.putText(frame, str(round(distance.item(), 2))+" "+str(condition_fashion), (person['box']['x1'], person['box']['y2']-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # threshold = 14
                        threshold = 12.5
                        print(f'Persona detectada en el fotograma. Distancia: {distance.item()}')
                        if distance.item() < threshold:
                        # if distance.item() < threshold and condition_fashion :
                        # if distance.item() <= threshold or condition_fashion :
                        # if distance.item() < threshold and (not condition_fashion or condition_fashion) :
                            distance_aux =  siamesas.euclidean_distance(reference_embedding, embedding_original)
                            if distance_aux.item() <= 12 and (not condition_fashion or condition_fashion) : 
                                print(distance_aux.item(), condition_fashion)
                                reference_embedding = siamesas.get_multimodal_embedding(model_s, frame_pil)
                                cv2.rectangle(frame, (person['box']['x1'], person['box']['y1']), (person['box']['x2'], person['box']['y2']), (0, 255, 0), 3)
                                cv2.putText(frame, f'{name}', (person['box']['x1'], person['box']['y2']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            else :
                                reference_embedding = embedding_original
                
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
                                if score > 0.50 :
                                    name_person = name.split('__')
                                    name_person = name_person[1] if len(name_person) > 1 else name_person
                                    cv2.putText(frame, f'{name_person}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    _, buffer = cv2.imencode('.jpg', frame)
                                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                    socketio.emit('person_recognized', {'person': name, 'frame': frame_base64})
                                    """ _, frame_person = cap.read()
                                    cv2.imwrite('./photo.png', frame_person) """
                                    # crop_people()

                                # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                                """ if matched_name != "Unknown" and score > 0.25 not in recognized_participants:
                                    recognized_participants.append(name)

                                    participant = Participant.query.filter_by(ci=name).first()
                                    
                                    if participant:
                                        existing_attendance = Attendance.query.filter_by(
                                            event_id=event_id, participant_id=participant.id
                                        ).first()
                                        if not existing_attendance:
                                            if event_id not in [event.id for event in participant.events]:
                                                event = Event.query.get(event_id)
                                                participant.events.append(event)
                                                socketio.emit('participant_recognized', {'ci': name})
                                                attendance = Attendance(event_id=event_id, participant_id=participant.id)
                                                db.session.add(attendance)
                                                db.session.commit() """

                        # cv2.putText(frame, f'{matched_name} - Score: {highest_score:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # cv2.imshow(f'Camera {camera_index}', frame)  # Muestra el cuadro de la cámara en una ventana.

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Permite cerrar la ventana presionando 'q'.
                    break

                time.sleep(0.01)  # Controla la velocidad del bucle.
        finally:
            cap.release()  # Libera la cámara al finalizar.
            cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV.

""" 
@app.route('/add_event')  # Define la ruta para mostrar el formulario de añadir un evento.
def add_event():  # Define la función que se ejecutará cuando se acceda a la ruta '/add_event'.
    return render_template('add_event.html')  # Renderiza la plantilla 'add_event.html' para mostrar el formulario al usuario.

@app.route('/save_event', methods=['POST'])  # Define la ruta para guardar un evento, utilizando el método POST.
def save_event():  # Define la función que se ejecutará cuando se acceda a la ruta '/save_event'.
    try:
        data = request.get_json()  # Obtiene los datos JSON enviados en la solicitud.
        nombre = data.get('nombre')  # Extrae el nombre del evento desde los datos.
        fecha = data.get('fecha')  # Extrae la fecha del evento desde los datos.
        hora = data.get('hora')  # Extrae la hora del evento desde los datos.

        if not nombre or not fecha or not hora:  # Verifica si alguno de los campos está vacío.
            abort(400, description="All fields are required.")  # Devuelve un error 400 si faltan campos.

        new_event = Event(nombre=nombre, fecha=fecha, hora=hora)  # Crea una nueva instancia del evento con los datos proporcionados.

        db.session.add(new_event)  # Añade el nuevo evento a la sesión de la base de datos.
        db.session.commit()  # Guarda los cambios en la base de datos.

        return jsonify({"message": "Event saved successfully."}), 200  # Devuelve una respuesta JSON indicando que el evento se guardó correctamente.

    except Exception as e:  # Captura cualquier excepción que ocurra durante el proceso.
        db.session.rollback()  # Revierte cualquier cambio en la sesión de la base de datos en caso de error.
        return jsonify({"error": str(e)}), 500  # Devuelve un error 500 con el mensaje de la excepción.


@app.route('/get_participants/<int:event_id>', methods=['GET'])  # Define la ruta para obtener los participantes de un evento específico, usando el método GET.
def get_participants(event_id):  # Define la función que se ejecutará cuando se acceda a la ruta '/get_participants/<event_id>'.
    try:
        # Obtener los IDs de los participantes asociados al evento usando la tabla intermedia
        participant_ids = db.session.query(participants_events.c.participant_id).filter_by(event_id=event_id).all()  # Consulta los IDs de los participantes vinculados al evento específico.

        # Extraer solo los IDs de los resultados
        participant_ids = [pid[0] for pid in participant_ids]  # Convierte los resultados de la consulta en una lista de IDs.

        # Obtener los detalles de los participantes utilizando los IDs obtenidos
        participants = db.session.query(Participant).filter(Participant.id.in_(participant_ids)).all()  # Consulta los detalles de los participantes que tienen los IDs obtenidos.

        # Convertir los resultados en una lista de diccionarios o cadenas, según lo que necesites
        participant_list = [{'id': p.id, 'ci': p.ci} for p in participants]  # Crea una lista de diccionarios con los IDs y CI de cada participante.

        return jsonify({'participants': participant_list})  # Devuelve la lista de participantes en formato JSON.
    except Exception as e:  # Captura cualquier excepción que ocurra durante el proceso.
        print(f"Error retrieving participants: {e}")  # Imprime un mensaje de error en la consola.
        return jsonify({'error': 'Unable to retrieve participants'}), 500  # Devuelve un error 500 con un mensaje indicando que no se pudieron recuperar los participantes.

    
@app.route('/show_participants/<int:event_id>')  # Define la ruta para mostrar los participantes de un evento específico.
def show_participants(event_id):  # Define la función que se ejecutará cuando se acceda a la ruta '/show_participants/<event_id>'.
    try:
        # Obtener los IDs de los participantes asociados al evento usando la tabla intermedia
        participant_ids = db.session.query(participants_events.c.participant_id).filter_by(event_id=event_id).all()  # Consulta los IDs de los participantes vinculados al evento específico.

        # Extraer solo los IDs de los resultados
        participant_ids = [pid[0] for pid in participant_ids]  # Convierte los resultados de la consulta en una lista de IDs.

        # Obtener los detalles de los participantes utilizando los IDs obtenidos
        participants = db.session.query(Participant).filter(Participant.id.in_(participant_ids)).all()  # Consulta los detalles de los participantes que tienen los IDs obtenidos.

        return render_template('show_participants.html', participants=participants)  # Renderiza la plantilla 'show_participants.html' pasando la lista de participantes como contexto.
    except Exception as e:  # Captura cualquier excepción que ocurra durante el proceso.
        print(f"Error retrieving participants: {e}")  # Imprime un mensaje de error en la consola.
        return "<p>Error al recuperar los participantes.</p>", 500  # Devuelve un mensaje de error simple en caso de que no se puedan recuperar los participantes.
 """

if __name__ == "__main__":  # Comprueba si el script se está ejecutando directamente.
    
    with app.app_context():  # Crea un contexto de aplicación para ejecutar el siguiente bloque de código.
        db.create_all()  # Crea todas las tablas en la base de datos según los modelos definidos.

    socketio.run(app, debug=False)  # Inicia la aplicación Flask con SocketIO, sin habilitar el modo de depuración.


