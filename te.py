from ultralytics import YOLO
import pickle, cv2, onnxruntime
import numpy as np

class Detect_YOLO():
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
    def predict(self,img, conf=0.45, device=0 ):
        # results = self.model.predict(source=img, imgsz=640, conf=conf, device=device, verbose=False)
        results = self.model.predict(source=img, imgsz=640, conf=conf, device='cpu', verbose=False)
        results = results[0].cpu().numpy()
        list_box = []
        if len(results.boxes):
            for box in results.boxes:
                list_box.append(box.xyxy[0])
        # print(list_box)
        sorted_boxes = sorted(list_box, key=lambda box: box[0])
        # return  min(list_box, key=lambda box: box[0]) 
        if len(sorted_boxes) >=5:    
            return [sorted_boxes[0], sorted_boxes[1], sorted_boxes[-1], sorted_boxes[-2], sorted_boxes[-3]]  
        else: 
            return list_box

class Extract_model:
    def __init__(self, model_file=None):
        assert model_file is not None
        self.model_file = model_file
        self.input_mean = 127.5
        self.input_std = 127.5
        self.session = onnxruntime.InferenceSession(self.model_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        self.output_shape = outputs[0].shape
    
    def get(self, img):
        embedding = self.get_feat(img).flatten()
        return embedding

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

class Huminity_reg():
    def __init__(self, model_yolo, model_extract, thre_distance):
        self.model_det = Detect_YOLO(model_yolo)
        self.model_extract = Extract_model(model_extract)
        with open('Model/CHUOI/data.pickle', 'rb') as f:
            self.register_data = pickle.load(f)
        self.thre_distance = thre_distance

    def compare(self, emb):
        distance = np.linalg.norm(self.register_data['embs'] - emb, axis=1)
        idx = np.argmin(distance)
        name = self.register_data['name'][idx]
        return name, min(distance)

    def recognize(self,img):
        list_box = self.model_det.predict(img, conf=0.2, device='cpu')
        list_output = []
        for box in list_box:
            x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # image = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            img_cr = img[y1:y2, x1:x2]
            emb = self.model_extract.get(img_cr)
            output, dis = self.compare(emb)
            list_output.append(output)
        return list_output



if __name__ == "__main__":
    img_path = "D:/DU LIEU_MAY MOI/DU LIEU CHUOI/CHUOI 02_24_05_2024/image_2024_05_25_17_16_31_Khay1-533__Khay2-468__Tong1001__NhietDo-49.123260498046875__DoAm-23.336227416992188.jpg"
    img = cv2.imread(img_path)
    ex = Huminity_reg(model_yolo="Model/CHUOI/best.pt", model_extract="Model/CHUOI/model_sim.onnx", thre_distance=6.4)
    list_output  = ex.recognize(img)
    sorted_data = sorted(list_output, key=lambda x: x[1])
    result = [key for key, value in sorted_data[:2]]
    print(sorted_data[0])