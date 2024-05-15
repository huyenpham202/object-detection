import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, classes, COLORS, class_id, confidences, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    confidence_percent = round(confidences * 100, 2)
    label_with_confidence = f"{label} {confidence_percent}%"
    (label_width, label_height), baseline = cv2.getTextSize(label_with_confidence, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x - 5, y - label_height - 5), (x + label_width + 5, y), color, -1)
    cv2.putText(img, label_with_confidence, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def browse_image():
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image Files", "*.jpg *.png *.jpeg"),))
    entry_image.delete(0, tk.END)
    entry_image.insert(0, filename)

def detect_image():
    image_path = entry_image.get()
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    classes_path = "yolov3.txt"

    # Kiểm tra xem đường dẫn ảnh có hợp lệ không
    if not image_path:
        messagebox.showerror("Error", "Vui lòng chọn ảnh để nhận diện!")
        return

    image = cv2.imread(image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(weights_path, config_path)
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.7
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
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
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, classes, COLORS, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    cv2.imshow("Anh sau khi phat hien", image)
    cv2.waitKey(0)

    cv2.imwrite("ketqua/AnhSauKhiPhatHien.jpg", image)
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Nhập ảnh cần nhận diện")

label_image = tk.Label(root, text="Đường dẫn ảnh:", font=("Arial", 14))
label_image.grid(row=0, column=0, padx=(10, 5), pady=20, sticky="e")

entry_image = tk.Entry(root, width=50, font=("Arial", 14))
entry_image.grid(row=0, column=1, columnspan=3, padx=5, pady=(20,30), sticky="ew")

button_browse = tk.Button(root, text="Browse", command=browse_image, font=("Arial", 14))
button_browse.grid(row=0, column=4, padx=5, pady=20)

button_detect = tk.Button(root, text="Nhận diện", command=detect_image, font=("Arial", 14), bg="green", fg="white")
button_detect.grid(row=1, column=1, columnspan=3, padx=5, pady=20, sticky="ew")
root.mainloop()
