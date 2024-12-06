from ultralytics import YOLO
import cv2
import ultralytics

ultralytics.checks()
# Load model
model = YOLO('ppe.pt')  # Đường dẫn đến file model

# Đọc ảnh
img_path = 'img.jpg'  # Đường dẫn đến ảnh cần detect
img = cv2.imread(img_path)

# Sử dụng model để detect
results = model(img)
result = results[0]

# Hiển thị kết quả
result.show()  # Hiển thị kết quả trên cửa sổ hình ảnh
# Lưu kết quả
result.save()  # Lưu kết quả vào thư mục output​10:39/-strong/-heart:>:o:-((:-h 11:27  01/12/2024