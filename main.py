import cv2
import numpy as np
import os


# Загрузка предобученной модели Yolo для обнаружения автомобиля на изображении
def load_yolo_model(cfg_path, weights_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    try:
        unconnected_layers = net.getUnconnectedOutLayers()
        if isinstance(unconnected_layers, np.ndarray):
            unconnected_layers = unconnected_layers.flatten()
    except AttributeError:
        unconnected_layers = [net.getUnconnectedOutLayers()]
    output_layers = [layer_names[i - 1] for i in unconnected_layers]
    return net, output_layers


def detect_vehicles(net, output_layers, image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # class_id == 2 для автомобилей
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], confidences[i]) for i in indices]


def filter_taxis(image, detected_vehicles):
    taxis = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower_yellow = np.array([15, 150, 150])
    # upper_yellow = np.array([35, 255, 255])

    # lower_yellow = np.array([15, 150, 170])
    # upper_yellow = np.array([25, 255, 255])

    lower_yellow = np.array([15,100,200], np.uint8)                 # Требует более тонкой настройки
    upper_yellow = np.array([35,255,255], np.uint8)

    for (box, confidence) in detected_vehicles:
        x, y, w, h = box
        # Проверка корректности координат ROI
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > image.shape[1]: w = image.shape[1] - x
        if y + h > image.shape[0]: h = image.shape[0] - y

        if w > 0 and h > 0:
            car_roi = hsv[y:y + h, x:x + w]
            mask = cv2.inRange(car_roi, lower_yellow, upper_yellow)
            if np.any(mask):
                taxis.append(box)
    return taxis


def detect_taxi(image_path, output_path, net, output_layers):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return

    detected_vehicles = detect_vehicles(net, output_layers, image)
    taxis = filter_taxis(image, detected_vehicles)

    for (x, y, w, h) in taxis:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, image)


if __name__ == "__main__":
    input_folder = "input_images"
    output_folder = "output_images"

    cfg_path = "yolov3.cfg"
    weights_path = "yolov3.weights"

    net, output_layers = load_yolo_model(cfg_path, weights_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            detect_taxi(os.path.join(input_folder, filename), output_folder, net, output_layers)

