import os
import tkinter as tk
from tkinter import Label, Frame, Canvas, Scrollbar, filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import h5py

# Tắt các cảnh báo của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Tải mô hình .h5
def load_model(model_path):
    with h5py.File(model_path, 'r') as f:
        model = tf.keras.models.load_model(f)
    return model

model = load_model('model.h5')

# Hàm tiền xử lý ảnh
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((128, 128))  # Kích thước ảnh mà mô hình yêu cầu
    image_array = np.array(image) / 255.0  # Chuẩn hóa giá trị ảnh
    image_array = np.expand_dims(image_array, axis=0)  # Thêm batch dimension
    return image_array

# Hàm dự đoán
def predict_image(image_path):
    image_array = preprocess_image(image_path)
    prediction = model.predict(image_array)
    return "Chó" if prediction[0][0] <= 0.5 else "Mèo"

def show_image_with_result(image_path, back_command):
    global panel, label_result, btn_back
    
    # Xóa các widget hiện tại trong main_frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Hiển thị ảnh
    image = Image.open(image_path)
    image = image.resize((300, 300))
    img = ImageTk.PhotoImage(image)
    panel = Label(main_frame, image=img)
    panel.image = img
    panel.pack()

    # Dự đoán và hiển thị kết quả
    result = predict_image(image_path)
    label_result = Label(main_frame, text=f"Kết quả: {result}")
    label_result.pack()

    # Hiển thị nút quay lại
    btn_back.config(command=back_command)
    btn_back.pack(side=tk.BOTTOM)

def display_images_in_folder(folder_path):
    global canvas, scrollable_frame

    # Xóa các widget hiện tại
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Tạo canvas và scrollbar
    canvas = Canvas(main_frame)
    scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    # Cấu hình scrollbar
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Hiển thị các hình ảnh trong folder
    row, col = 0, 0
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            image = image.resize((200, 200))  # Tăng kích thước lên gấp đôi
            img = ImageTk.PhotoImage(image)
            btn = tk.Button(scrollable_frame, image=img, command=lambda p=image_path: show_image_with_result(p, lambda: display_images_in_folder(folder_path)))
            btn.image = img  # Giữ tham chiếu đến ảnh để không bị garbage collected
            btn.grid(row=row, column=col, padx=10, pady=10)  # Tăng khoảng cách giữa các ảnh
            col += 1
            if col == 3:  # 3 hình ảnh mỗi hàng
                col = 0
                row += 1

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

def back_to_main_interface():
    global btn_select_folder
    
    # Xóa các widget hiện tại
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Tạo lại các widget chính
    btn_select_folder = tk.Button(main_frame, text="Chọn thư mục hình ảnh", command=lambda: display_images_in_folder(filedialog.askdirectory()))
    btn_select_folder.pack()

# Khởi tạo cửa sổ chính
root = tk.Tk()
root.title("Image Classifier")
root.geometry("800x600")

main_frame = Frame(root)
main_frame.pack(fill="both", expand=True)

# Nút chọn thư mục chứa hình ảnh với kích thước gấp đôi
btn_select_folder = tk.Button(main_frame, text="Chọn thư mục hình ảnh", command=lambda: display_images_in_folder(filedialog.askdirectory()), height=2, width=20)
btn_select_folder.pack()

# Panel để hiển thị ảnh và nhãn kết quả
panel = Label(main_frame)
panel.pack()

# Nhãn để hiển thị kết quả dự đoán
label_result = Label(main_frame, text="Kết quả:")
label_result.pack()

# Nút quay lại với kích thước gấp đôi
btn_back = tk.Button(root, text="Quay lại", command=back_to_main_interface, height=2, width=20)

# Chạy giao diện
root.mainloop()