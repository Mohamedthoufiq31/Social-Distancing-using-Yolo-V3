import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet('yolo/yolov3.weights', 'yolo/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open('yolo/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load Image
image_path = 'your_image.jpg'  # Path to your image file
image = cv2.imread(image_path)
height, width, channels = image.shape

# Detecting Objects
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Show Information on Screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id == 0:  # Only consider the "person" class
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Define the distance threshold (in pixels)
distance_threshold = 75

def calculate_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center_x1, center_y1 = x1 + w1 / 2, y1 + h1 / 2
    center_x2, center_y2 = x2 + w2 / 2, y2 + h2 / 2
    distance = np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
    return distance

# Draw bounding boxes and check social distancing
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        color = (0, 255, 0)
        for j in range(len(boxes)):
            if j in indexes and i != j:
                distance = calculate_distance(boxes[i], boxes[j])
                if distance < distance_threshold:
                    color = (0, 0, 255)
                    break
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

cv2.imshow('Social Distancing Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
