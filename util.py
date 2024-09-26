
from ultralytics import YOLO
model_person = YOLO('./models/model_people.pt')

class Util :
    
    def get_people(self, path) :
        import torch
        modelYolo = torch.hub.load('./models/yolov5', 'custom', './models/model_people.pt', source='local')
        result = modelYolo(path)
        return result
    
    def get_peoplev8(self, path) :
        from ultralytics import YOLO
        model = YOLO('./models/model_people2.pt')
        return model(path, stream=False)
    
    def get_fashion(self, path) :
        from ultralytics import YOLO
        model_fashion = YOLO('./models/model_fashion.pt')
        return model_fashion(path, stream=False)
