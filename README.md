# 🛡️ Ứng Dụng Web Phát Hiện Giao Dịch Gian Lận Tài Chính (Streamlit)

Ứng dụng web tương tác trực quan được phát triển trên nền tảng **Streamlit** giúp chuyển đổi quy trình thử nghiệm từ Notebook phân tích dữ liệu thành giải pháp phần mềm quản trị rủi ro hoàn chỉnh.

## ✨ Tính năng nổi bật

1. **Tổng Quan Dữ Liệu (EDA):** Tự động phân tích cấu trúc, thống kê chi tiết đặc trưng và hiển thị tỷ lệ mất cân bằng của biến mục tiêu `default`.
2. **Tiền Xử Lý Linh Hoạt:** Cho phép cấu hình trực tiếp tỷ lệ chia tách tập huấn luyện (Train) và tập kiểm thử (Test) thông qua giao diện kéo trượt.
3. **Huấn Luyện Đa Thuật Toán:** Triển khai đồng thời 3 mô hình phân lớp mạnh mẽ bao gồm `Logistic Regression`, `Decision Tree` và `Random Forest`.
4. **Đánh Giá Trực Quan:** So sánh toàn diện hiệu năng của các mô hình dựa trên các chỉ số nâng cao chuyên biệt: `Accuracy`, `Precision`, `Recall`, `F1-Score` phối hợp cùng biểu đồ ma trận nhầm lẫn (Confusion Matrix).
5. **Dự Báo Đa Chế Độ:**
   - **Nhập dữ liệu trực tiếp:** Điền thông tin nhanh cho từng hồ sơ khách hàng đơn lẻ để chấm điểm thời gian thực.
   - **Xử lý tệp lớn (Batch Processing):** Tải file Excel/CSV chứa hàng loạt giao dịch mới, hệ thống tự động kiểm định và cho phép xuất/tải file báo cáo kết quả đã dán nhãn rủi ro.

## 🚀 Hướng dẫn cài đặt và chạy ứng dụng

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã cài đặt sẵn môi trường **Python (phiên bản >= 3.9)**.

### Bước 2: Cài đặt các thư viện phụ thuộc
Di chuyển dòng lệnh vào thư mục chứa dự án và thực thi lệnh cài đặt từ file `requirements.txt`:
```bash
pip install -r requirements.txt
