# 📊 Kết Quả Đánh Giá Model J-48

## 📁 Các File Trong Thư Mục Này

### 1. `classification_report.txt`
Báo cáo chi tiết về hiệu suất của model, bao gồm:
- **Metrics Summary**: Tổng hợp các chỉ số chính (Accuracy, Precision, Recall, F1-Score)
- **Detailed Classification Report**: Báo cáo chi tiết cho từng lớp
- **Confusion Matrix**: Ma trận nhầm lẫn hiển thị số lượng dự đoán đúng/sai

### 2. `confusion_matrix.json`
Ma trận nhầm lẫn dạng JSON, bao gồm:
- `confusion_matrix`: Ma trận 2D hiển thị số lượng dự đoán
- `classes`: Danh sách các lớp target
- Các metrics: accuracy, precision, recall, f1-score

### 3. `metrics_summary.json`
Tổng hợp các metrics chính của model:
- `accuracy`: Độ chính xác tổng thể (0.975 = 97.5%)
- `precision_weighted`: Precision trung bình có trọng số
- `recall_weighted`: Recall trung bình có trọng số
- `f1_weighted`: F1-Score trung bình có trọng số
- `target_classes`: Danh sách các lớp được dự đoán
- `feature_columns`: Danh sách các features sử dụng

## 📈 Kết Quả Hiện Tại

- **Accuracy**: 97.50% - Rất tốt! ✅
- **Precision (weighted)**: 97.56%
- **Recall (weighted)**: 97.50%
- **F1-Score (weighted)**: 97.52%

### Phân tích theo từng lớp:

| Lớp | Precision | Recall | F1-Score | Support |
|-----|-----------|--------|----------|---------|
| Critically Endangered | 0.98 | 0.98 | 0.98 | 55 |
| Data Deficient | 0.95 | 0.91 | 0.93 | 23 |
| Endangered | 0.83 | 0.91 | 0.87 | 11 |
| Least Concern | 1.00 | 1.00 | 1.00 | 77 |
| Vulnerable | 0.97 | 0.97 | 0.97 | 34 |

## 💡 Giải Thích Metrics

### Accuracy (Độ chính xác)
Tỷ lệ dự đoán đúng trên tổng số mẫu test.
- **97.50%** nghĩa là trong 200 mẫu test, model dự đoán đúng 195 mẫu.

### Precision (Độ chính xác dự đoán)
Tỷ lệ các dự đoán dương tính thực sự là dương tính.
- **97.56%** nghĩa là trong các dự đoán của model, 97.56% là đúng.

### Recall (Độ nhạy)
Tỷ lệ các mẫu dương tính thực tế được model tìm thấy.
- **97.50%** nghĩa là model tìm thấy 97.50% các mẫu dương tính thực tế.

### F1-Score
Trung bình điều hòa của Precision và Recall, cân bằng giữa hai chỉ số.
- **97.52%** cho thấy model cân bằng tốt giữa Precision và Recall.

## 🎯 Kết Luận

Model J-48 đạt được hiệu suất **rất tốt** với độ chính xác **97.50%**. Model có thể được sử dụng để dự đoán tình trạng bảo tồn của cá sấu một cách đáng tin cậy.

### Điểm mạnh:
- ✅ Độ chính xác cao (97.50%)
- ✅ Precision và Recall đều cao và cân bằng
- ✅ Dự đoán tốt cho hầu hết các lớp

### Điểm cần cải thiện:
- ⚠️ Lớp "Endangered" có Precision thấp hơn (0.83) - có thể do số lượng mẫu ít (11 mẫu)

---

**Ngày tạo:** 2025
**Model:** J-48 (Decision Tree)
**Criterion:** entropy


