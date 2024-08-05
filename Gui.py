from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog
from PyQt5.uic import loadUi
from ultralytics import YOLO
import warnings, re, cv2, os, pickle, onnxruntime
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)




file_path = "ID_Cameras.txt"

def is_numeric_string(string):
    for char in string:
        if not char.isdigit():
            return False
    return True

def contains_ip_pattern(string):
    pattern = r"192\.168\.\d+\.\d+"
    if re.search(pattern, string):
        return True
    return False



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

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write("")

        self.is_webcam_on = None
        self.capture = None
        self.ui_files = {
            'Gui_Start': {'file': 'File_UI/Start_UI.ui', 'width': 1130, 'height': 650},
            'Gui_Camera': {'file': 'File_UI/UI_Camera.ui', 'width': 1130, 'height': 650},
            'Gui_Image': {'file': 'File_UI/UI_Image.ui', 'width': 1130, 'height': 650},
            'Gui_Change_Camera': {'file': 'File_UI/UI_Change_Camera.ui', 'width': 772, 'height': 150}
        }
        self.load_ui('Gui_Start')
        self.actionCamera.triggered.connect(self.Camera)
        self.actionImage.triggered.connect(self.Image)
        self.actionChange_Camera.triggered.connect(self.Change_Camera)
        self.is_webcam_on = False
    
    def load_ui(self, ui_name):
        ui_info = self.ui_files.get(ui_name)
        if ui_info:
            file_paths = ui_info['file']
            width = ui_info['width']
            height = ui_info['height']
            # Load giao diện từ file .ui
            loadUi(file_paths, self)
            # Đặt kích thước cố định cho cửa sổ chương trình
            self.setFixedSize(width, height)

            if ui_name == 'Gui_Start':
                self.display_images()

    def display_images(self):
        image_names = ['Image_1', 'Image_2', 'Image_3', 'Image_4', 'Image_5']
        image_files = ['Chuoi.jpg', 'Man.jpg', 'Xoai.jpg', 'Cam.jpg', 'dua (2).jpg']
        image_dir = "Image"  # Thư mục chứa các ảnh
        
        for qframe_name, image_file in zip(image_names, image_files):
            qframe = self.findChild(QtWidgets.QFrame, qframe_name)
            if qframe:
                image_path = os.path.join(image_dir, image_file)
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    
                    # Tạo pixmap từ QImage và thay đổi kích thước theo QFrame
                    pixmap = QPixmap.fromImage(qimage)
                    pixmap = pixmap.scaled(qframe.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                    
                    label = QtWidgets.QLabel(qframe)
                    label.setPixmap(pixmap)
                    label.setScaledContents(True)
                    label.setGeometry(qframe.rect())
                    
                    layout = QtWidgets.QVBoxLayout(qframe)
                    layout.addWidget(label)
                    qframe.setLayout(layout)
                else:
                    qframe.setVisible(False)
    
    def closeEvent(self, event):
        """Override closeEvent to confirm window close."""
        reply = QMessageBox.question(self, "Xác nhận thoát", "Bạn có chắc chắn muốn thoát?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("Bạn đã thoát")
            event.accept()
            self.is_webcam_on = False
            self.close()  # Close the window
        else:
            event.ignore()
    
    def Start_Camera(self):
        with open(file_path, "r") as file:
            ID_Camera = file.read()
        if not ID_Camera:
            QMessageBox.warning(self, "Cảnh báo", "Không tìm thấy ID Camera hoặc giá trị rỗng.", QMessageBox.Ok)
            self.Change_Camera()
            return
        if is_numeric_string(ID_Camera):
            self.capture = cv2.VideoCapture(int(ID_Camera))
        if contains_ip_pattern(ID_Camera):
            self.capture = cv2.VideoCapture(str(ID_Camera))
        self.is_webcam_on = True
        self.is_webcam_on = self.capture.isOpened()
        if self.is_webcam_on:
            while self.is_webcam_on:
                ret, frame = self.capture.read()
                if ret:
                    frame = cv2.resize(frame, [1120, 820])
                    # cv2.imwrite("x.jpg", frame)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(image, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(image)
                    self.Show_Camera.setPixmap(pixmap)


                    try:
                        ex = Huminity_reg(model_yolo="Model/CHUOI/best.pt", model_extract="Model/CHUOI/model_sim.onnx", thre_distance=6.4)
                        list_output  = ex.recognize(frame)
                        sorted_data = sorted(list_output, key=lambda x: x[1])
                        result = [key for key, value in sorted_data[:2]]
                        print(sorted_data[0])
                        self.KQ_Camera.setText(f"Giá trị độ ẩm dự đoán: {sorted_data[0]}%")
                    except:
                        self.KQ_Camera.setText("Không thể dự đoán")


                    QApplication.processEvents()
            self.capture.release()
        else:
            QMessageBox.information(self, "Lỗi", "Chưa kết nối với Camera", QMessageBox.Ok)

    def Start_Image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            image_path = selected_files[0]
            image = cv2.imread(image_path)
            try:
                ex = Huminity_reg(model_yolo="Model/CHUOI/best.pt", model_extract="Model/CHUOI/model_sim.onnx", thre_distance=6.4)
                list_output  = ex.recognize(image)
                sorted_data = sorted(list_output, key=lambda x: x[1])
                result = [key for key, value in sorted_data[:2]]
                print(sorted_data[0])
                self.KQ_Image.setText(f"Giá trị độ ẩm dự đoán: {sorted_data[0]}%")
            except:
                self.KQ_Image.setText("Không thể dự đoán")




            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.Show_Image.setPixmap(pixmap)
            self.Show_Image.setScaledContents(True)

    def Change_ID_Camera(self):
        # print("Xác nhận bạn đã nhấn nút thay đổi ID Camera!")
        ID_Camera = self.ID_Camera.toPlainText()
        # print(ID_Camera)
        with open(file_path, "w") as file:
            file.write(ID_Camera)
        self.Camera()
        # print(f"Đã lưu giá trị {ID_Camera} vào file ID_Cameras.txt")
    
    def Camera(self):
        # print("Đã gọi đến camera")
        self.load_ui('Gui_Camera')
        self.actionImage.triggered.connect(self.Image)
        self.actionChange_Camera.triggered.connect(self.Change_Camera)
        self.btn_start_camera.clicked.connect(self.Start_Camera)
    
    def Image(self):
        # print("Đã gọi đến Image")
        self.is_webcam_on = False
        self.load_ui('Gui_Image')
        self.actionCamera.triggered.connect(self.Camera)
        self.actionChange_Camera.triggered.connect(self.Change_Camera)
        self.btn_search.clicked.connect(self.Start_Image)
    
    def Change_Camera(self):
        # print("Đã gọi đến thay đổi Camera")
        self.is_webcam_on = False
        self.load_ui('Gui_Change_Camera')
        self.actionCamera.triggered.connect(self.Camera)
        self.actionImage.triggered.connect(self.Image)
        with open(file_path, "r") as file:
            ID_Camera = file.read()
            self.ID_Camera.setPlainText(ID_Camera)
        self.btn_Change_ID.clicked.connect(self.Change_ID_Camera)



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ex = Ui_MainWindow()
    ex.show()
    sys.exit(app.exec_())