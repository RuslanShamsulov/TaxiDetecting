import os
import cv2
from main import detect_taxi, load_yolo_model

# Тестовый скрипт

def test_detect_taxi():
    input_folder = "test_input"
    output_folder = "test_output"
    cfg_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    net, output_layers = load_yolo_model(cfg_path, weights_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            detect_taxi(input_path, output_folder, net, output_layers)

            assert os.path.exists(output_path), f"Output file {output_path} does not exist"

            # Проверка наличия рамок
            output_image = cv2.imread(output_path)
            gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            assert len(contours) > 0, f"No contours found in {output_path}"


if __name__ == "__main__":
    test_detect_taxi()