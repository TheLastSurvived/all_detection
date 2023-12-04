from flask import Flask, render_template, Response, request
import cv2
import serial
import threading
import time
import json
import argparse
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # веб камера

controlX, controlY = 0, 0  # глобальные переменные положения джойстика с web-страницы

cap = cv2.VideoCapture(0)

output_frames_per_second = 20.0
 
RESIZED_DIMENSIONS = (300, 300) # Dimensions that SSD was trained on. 
IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255
 
# Load the pre-trained neural network
neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
        'MobileNetSSD_deploy.caffemodel')

# Load the pre-trained neural network
neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
        'MobileNetSSD_deploy.caffemodel')
 
# List of categories and classes
categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
               4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
               9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
              13: 'horse', 14: 'motorbike', 15: 'person', 
              16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
              19: 'train', 20: 'tvmonitor'}
 
classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", 
           "diningtable",  "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
                      
# Create the bounding boxes
bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))


def getFramesGenerator():
    """ Генератор фреймов для вывода в веб-страницу, тут же можно поиграть с openCV"""
    while True:
            
            # Capture one frame at a time
        success, frame = cap.read() 
    
        # Do we have a video frame? If true, proceed.
        if success:
            
            # Capture the frame's height and width
            (h, w) = frame.shape[:2]
        
            # Create a blob. A blob is a group of connected pixels in a binary 
            # frame that share some common property (e.g. grayscale value)
            # Preprocess the frame to prepare it for deep learning classification
            frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS), 
                            IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
            
            # Set the input for the neural network
            neural_network.setInput(frame_blob)
        
            # Predict the objects in the image
            neural_network_output = neural_network.forward()
        
            # Put the bounding boxes around the detected objects
            for i in np.arange(0, neural_network_output.shape[2]):
                    
                confidence = neural_network_output[0, 0, i, 2]
            
                # Confidence must be at least 30%       
                if confidence > 0.30:
                        
                    idx = int(neural_network_output[0, 0, i, 1])
            
                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                        [w, h, w, h])
            
                    (startX, startY, endX, endY) = bounding_box.astype("int")
            
                    label = "{}: {:.2f}%".format(classes[idx], confidence * 100) 
                    
                    cv2.rectangle(frame, (startX, startY), (
                        endX, endY), bbox_colors[idx], 2)     
                                    
                    y = startY - 15 if startY - 15 > 15 else startY + 15    
            
                    cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, bbox_colors[idx], 2)
            
        # We now need to resize the frame so its dimensions
        # are equivalent to the dimensions of the original frame
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """ Генерируем и отправляем изображения с камеры"""
    return Response(getFramesGenerator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """ Крутим html страницу """
    return render_template('index.html')


@app.route('/control')
def control():
    """ Пришел запрос на управления роботом """
    global controlX, controlY
    controlX, controlY = float(request.args.get('x')) / 100.0, float(request.args.get('y')) / 100.0
    return '', 200, {'Content-Type': 'text/plain'}


if __name__ == '__main__':
    # пакет, посылаемый на робота
    msg = {
        "speedA": 0,  # в пакете посылается скорость на левый и правый борт тележки
        "speedB": 0  #
    }

    # параметры робота
    speedScale = 0.65  # определяет скорость в процентах (0.50 = 50%) от максимальной абсолютной
    maxAbsSpeed = 100  # максимальное абсолютное отправляемое значение скорости
    sendFreq = 10  # слать 10 пакетов в секунду

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5000, help="Running port")
    parser.add_argument("-i", "--ip", type=str, default='127.0.0.1', help="Ip address")
    parser.add_argument('-s', '--serial', type=str, default='/dev/ttyUSB0', help="Serial port")
    args = parser.parse_args()

    serialPort = serial.Serial(args.serial, 9600)   # открываем uart

    def sender():
        """ функция цикличной отправки пакетов по uart """
        global controlX, controlY
        while True:
            speedA = maxAbsSpeed * (controlY + controlX)    # преобразуем скорость робота,
            speedB = maxAbsSpeed * (controlY - controlX)    # в зависимости от положения джойстика

            speedA = max(-maxAbsSpeed, min(speedA, maxAbsSpeed))    # функция аналогичная constrain в arduino
            speedB = max(-maxAbsSpeed, min(speedB, maxAbsSpeed))    # функция аналогичная constrain в arduino

            msg["speedA"], msg["speedB"] = speedScale * speedA, speedScale * speedB     # урезаем скорость и упаковываем

            serialPort.write(json.dumps(msg, ensure_ascii=False).encode("utf8"))  # отправляем пакет в виде json файла
            time.sleep(1 / sendFreq)

    threading.Thread(target=sender, daemon=True).start()    # запускаем тред отправки пакетов по uart с демоном

    app.run(debug=False, host=args.ip, port=args.port)   # запускаем flask приложение
