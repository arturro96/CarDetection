import cv2
import numpy as np


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]

    car_sub_img = img[y:y_plus_h, x:x_plus_w]

    plates = car_cascade.detectMultiScale(car_sub_img, 1.1, 1)
    for (p_x, p_y, p_w, p_h) in plates:
        plate_img = car_sub_img[p_y:p_y + p_h, p_x:p_x + p_w]
        plate_img = cv2.GaussianBlur(plate_img, (23, 23), 30)
        img[y + p_y:y + p_y + plate_img.shape[0], x + p_x:x + p_x + plate_img.shape[1]] = plate_img

    cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), color, 2)
    cv2.putText(img, label, (int(x - 10), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)


CLASSES_FILE = "model/yolov3.txt"
MODEL_CONFIG_FILE = "model/yolov3.cfg"
MODEL_WEIGHTS_FILE = "model/yolov3.weights"

VEHICLES_CLASS_IDS = [2, 5, 6, 7]

with open(CLASSES_FILE, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(MODEL_WEIGHTS_FILE, MODEL_CONFIG_FILE)
car_cascade = cv2.CascadeClassifier('russianplate.xml')

cap = cv2.VideoCapture('plates.mp4')

while True:
    ret, image = cap.read()

    if ret:

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)

                if class_id not in VEHICLES_CLASS_IDS:
                    continue

                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))

        cv2.imshow("object detection", image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break


cv2.destroyAllWindows()
