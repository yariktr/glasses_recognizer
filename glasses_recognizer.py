import onnxruntime as ort
import numpy
import cv2
import os
import shutil
import tkinter as tk
from tkinter import filedialog


global_directory_path = ""

def choose_directory():
    global global_directory_path
    directory_path = filedialog.askdirectory()
    if directory_path:
        global_directory_path = directory_path
        path_label.config(text='Вы выбрали: ' + directory_path)

def lets_go():
    if global_directory_path == "":
        path_label.config(text='Сначала выберете директорию!')
    else:
        root.destroy()

root = tk.Tk()
root.title("Glasses recognizer")
root.geometry('800x400')

welcome_label = tk.Label(root, text='Вас приветствует детектор очков на фото!\nЧтобы начать анализ изображений, выберите соответствующую директорию,\nа затем нажмите "Поехали!"', font=('Helvetica', 16), fg='black')
welcome_label.pack(pady=20)  

button_frame = tk.Frame(root)
button_frame.pack(pady=80)

browse_button = tk.Button(button_frame, text="Выбрать директорию", command=choose_directory,
                          font=('Helvetica', 12, 'bold'), bg='blue', fg='white', padx=10, pady=5)
browse_button.grid(row=0, column=0, padx=5, pady=10)

letsgo_button = tk.Button(button_frame, text="Поехали!", command=lets_go,
                         font=('Helvetica', 12, 'bold'), bg='red', fg='white', padx=10, pady=5)
letsgo_button.grid(row=0, column=1, padx=5, pady=10)


path_label = tk.Label(root, text="", wraplength=300,
                      font=('Helvetica', 10, 'italic'), bg='lightgray', fg='black', padx=10, pady=10)
path_label.pack(pady=10, fill=tk.BOTH, expand=True)


root.mainloop()


def resize(image, input_size):
    shape = image.shape
    ratio = float(shape[0]) / shape[1]
    if ratio > 1:
        h = input_size
        w = int(h / ratio)
    else:
        w = input_size
        h = int(w * ratio)
    scale = float(h) / shape[0]
    resized_image = cv2.resize(image, (w, h))
    det_image = numpy.zeros((input_size, input_size, 3), dtype=numpy.uint8)
    det_image[:h, :w, :] = resized_image
    return det_image, scale


ort_session = ort.InferenceSession('last.onnx')
input_size = 640
directory_path = global_directory_path
directory_path = directory_path.replace('\\','/')

processed_images_path = directory_path + '/processed_images'
if os.path.exists(processed_images_path):
    shutil.rmtree(processed_images_path)
os.makedirs(processed_images_path)


for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    file_path = file_path.replace('\\','/')
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        original_image = cv2.imread(file_path)
        image = original_image.copy()
        if image is not None:
            image, scale = resize(image, input_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))[::-1]
            image = image.astype('float32') / 255
            image = image[numpy.newaxis, ...]
            
            inputs = {ort_session.get_inputs()[0].name: image}
            outputs = ort_session.run(None, inputs)
            outputs_array = outputs[0][0]

            boxes = []
            scores = []
            indices = []
            for i in range(len(outputs_array)):
                x, y, w, h = outputs_array[i][0], outputs_array[i][1], outputs_array[i][2], outputs_array[i][3]
                left = int((x - w / 2) / scale)
                top = int((y - h / 2) / scale)
                width = int(w / scale)
                height = int(h / scale)
                scores.append(outputs_array[i][4])
                boxes.append([left, top, width, height])

            confidence_threshold = 0.5
            iou_threshold = 0.5

            indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)

            for indec in indices:
                x_center, y_center, width, height, confidence, class_confidence = outputs_array[indec]
                x = int((x_center - width / 2) / scale)
                y = int((y_center - height / 2) / scale)
                w = int(width / scale)
                h = int(height / scale)

                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                label = f'confidence: {confidence:.2f}, glasses'
                cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            
            output_img_path = os.path.join(processed_images_path, filename)
            output_img_path = output_img_path.replace('\\','/')
            cv2.imwrite(output_img_path, original_image)
            

processed_images_path = processed_images_path.replace('/','\\')  
        
root = tk.Tk()
root.title("Glasses recognizer")
root.geometry('600x300')

final_label = tk.Label(root, text='Все прошло без неприятностей!\nВы можете посмотреть обработанные изображения\nпо данному пути: ' + processed_images_path, font=('Helvetica', 16), fg='black')
final_label.pack(pady=70)  

root.mainloop()