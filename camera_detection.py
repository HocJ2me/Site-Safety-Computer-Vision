from ultralytics import YOLO
import cv2
import ultralytics

ultralytics.checks()

# Load model
model = YOLO('ppe.pt')  # Đường dẫn đến file model

# Mở camera (0 là camera mặc định)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc khung hình từ camera!")
        break

    # Sử dụng model để detect
    results = model(frame)
    result = results[0]

    # Hiển thị kết quả
    # result.show()  # Hiển thị kết quả trên cửa sổ hình ảnh
    # Lấy ảnh kết quả (có bounding box) và hiển thị
    img_with_boxes = result.plot()  # Lấy ảnh với các bounding box đã vẽ

    # Hiển thị kết quả
    cv2.imshow('Detection Result', img_with_boxes)  # Hiển thị ảnh với bounding boxes

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
