import cv2
import matplotlib.pyplot as plt

# === Config & Model Files ===
config_file = 'C:/Users/ASUS/Desktop/Face_detect/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt'
frozen_model = 'C:/Users/ASUS/Desktop/Face_detect/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'

# === Load the model ===
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# === Load class labels from coco.names ===
class_file = 'C:/Users/ASUS/Desktop/Face_detect/coco.names.txt'
with open(class_file, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

# === Set input parameters ===
model.setInputSize(120, 120)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# === Load image ===
img = cv2.imread('C:/Users/ASUS/Desktop/Face_detect/test.png')
plt.imshow(img)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()


# === Perform detection ===
classindex, confidence, bbox = model.detect(img, confThreshold=0.5)
print("Detected class indices:", classindex)

for i, name in enumerate(classlabels):
    print(f"{i+1}: {name}")


# === Drawing bounding boxes and labels ===
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 1.5

if len(classindex) != 0:
    for classind, conf, box in zip(classindex.flatten(), confidence.flatten(), bbox):
        label = f"{classlabels[classind - 1]}: {round(conf * 100, 2)}%"
        cv2.rectangle(img, box, color=(255, 0, 255), thickness=2)
        cv2.putText(img, label, (box[0] + 10, box[1] + 30), font, font_scale, (255, 255, 0), thickness=2)

# === Display image with detections ===
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detections")
plt.axis('off')
plt.show()
