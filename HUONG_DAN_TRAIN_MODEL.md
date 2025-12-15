# 📚 Hướng Dẫn Train Model J-48 cho Dự Đoán Tình Trạng Bảo Tồn Cá Sấu

## 📋 Mục Lục

1. [Giới thiệu](#giới-thiệu)
2. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
3. [Cài đặt môi trường](#cài-đặt-môi-trường)
4. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ liệu)
5. [Giải thích thuật toán J-48](#giải-thích-thuật-toán-j-48)
6. [Quy trình train model](#quy-trình-train-model)
7. [Giải thích từng bước](#giải-thích-từng-bước)
8. [Đánh giá kết quả](#đánh-giá-kết-quả)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Giới thiệu

Model J-48 là một thuật toán Decision Tree (Cây quyết định) được sử dụng để dự đoán tình trạng bảo tồn của cá sấu dựa trên các đặc điểm như chiều dài, cân nặng, nhóm tuổi, giới tính, quốc gia, môi trường sống và khu vực địa lý.

### Mục tiêu
- Dự đoán tình trạng bảo tồn của cá sấu với độ chính xác cao
- Phân loại thành 5 lớp: Critically Endangered, Data Deficient, Endangered, Least Concern, Vulnerable

---

## 💻 Yêu cầu hệ thống

### Phần mềm cần thiết:
- **Python**: 3.8 trở lên (khuyến nghị 3.10-3.13)
- **pip**: Package manager cho Python

### Thư viện Python:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.4.0
- joblib >= 1.3.0

---

## 🔧 Cài đặt môi trường

### Bước 1: Kiểm tra Python
```bash
python --version
```

### Bước 2: Cài đặt các thư viện

**Cách 1: Cài đặt từ requirements.txt**
```bash
pip install -r requirements.txt
```

**Cách 2: Cài đặt từng thư viện**
```bash
pip install pandas numpy scikit-learn joblib
```

**Lưu ý với Python 3.13:**
Nếu gặp lỗi khi cài đặt, sử dụng wheel có sẵn:
```bash
python -m pip install --only-binary :all: scikit-learn pandas numpy joblib
```

---

## 📊 Chuẩn bị dữ liệu

### File dữ liệu cần có:
- `crocodile_processed_complete.csv`: File CSV chứa dữ liệu đã được tiền xử lý

### Cấu trúc dữ liệu:
File CSV phải chứa các cột sau:
- `Observed Length (m)`: Chiều dài quan sát được (mét)
- `Observed Weight (kg)`: Cân nặng quan sát được (kilogram)
- `Age Class`: Nhóm tuổi (Hatchling, Juvenile, Subadult, Adult)
- `Sex`: Giới tính (Male, Female)
- `Country/Region`: Quốc gia/Khu vực
- `Habitat Type`: Loại môi trường sống
- `Continent`: Khu vực địa lý
- `Conservation Status`: Tình trạng bảo tồn (Target variable)

---

## 🌳 Giải thích thuật toán J-48

### J-48 là gì?
J-48 là một thuật toán Decision Tree được phát triển trong Weka (một công cụ machine learning). Trong scikit-learn, J-48 được triển khai bằng `DecisionTreeClassifier` với các tham số tương đương.

### Nguyên lý hoạt động:

1. **Cây quyết định**: Xây dựng một cây nhị phân, mỗi node đại diện cho một điều kiện kiểm tra trên một feature
2. **Entropy**: Sử dụng entropy để đo độ "hỗn loạn" của dữ liệu tại mỗi node
3. **Information Gain**: Chọn feature có Information Gain cao nhất để phân chia dữ liệu
4. **Đệ quy**: Lặp lại quá trình cho đến khi đạt điều kiện dừng

### Công thức Entropy:
```
Entropy(S) = -Σ p(i) * log₂(p(i))
```
Trong đó:
- S: Tập dữ liệu
- p(i): Tỷ lệ của lớp i trong tập dữ liệu

### Information Gain:
```
IG(S, A) = Entropy(S) - Σ (|Sv|/|S|) * Entropy(Sv)
```
Trong đó:
- A: Feature được chọn để phân chia
- Sv: Tập con sau khi phân chia theo feature A

### Ưu điểm:
- ✅ Dễ hiểu và giải thích
- ✅ Không cần chuẩn hóa dữ liệu (nhưng script này vẫn chuẩn hóa để tối ưu)
- ✅ Xử lý được cả dữ liệu số và phân loại
- ✅ Tự động chọn features quan trọng

### Nhược điểm:
- ⚠️ Dễ bị overfitting nếu cây quá sâu
- ⚠️ Nhạy cảm với dữ liệu nhiễu

---

## 🚀 Quy trình train model

### Bước 1: Chạy script train
```bash
python train_model.py
```

Hoặc trên Windows:
```bash
train_model.bat
```

### Bước 2: Kiểm tra kết quả
Sau khi train xong, các file sau sẽ được tạo:
- `model_j48.pkl`: Model đã được train
- `encoders/`: Thư mục chứa các encoder và scaler
- `model_metadata.json`: Thông tin về model
- `mappings.json`: Mapping các giá trị
- `result/`: Thư mục chứa kết quả đánh giá

---

## 📖 Giải thích từng bước

### Bước 1: Đọc dữ liệu
```python
df = pd.read_csv('crocodile_processed_complete.csv')
```
- Đọc file CSV vào DataFrame pandas
- Kiểm tra số lượng dòng và các cột

### Bước 2: Chuẩn bị dữ liệu
```python
df = df.dropna()  # Xóa các dòng có giá trị thiếu
X = df[feature_cols]  # Features (đầu vào)
y = df[target_col]    # Target (đầu ra)
```
- Loại bỏ dữ liệu thiếu để đảm bảo chất lượng
- Tách features và target

### Bước 3: Mã hóa dữ liệu

#### Mã hóa Age Class:
```python
age_mapping = {'Hatchling': 0, 'Juvenile': 1, 'Subadult': 2, 'Adult': 3}
X['Age Class'] = X['Age Class'].map(age_mapping)
```
- Chuyển đổi nhóm tuổi từ text sang số
- Hatchling (mới nở) = 0
- Juvenile (non trẻ) = 1
- Subadult (gần trưởng thành) = 2
- Adult (trưởng thành) = 3

#### Mã hóa các biến phân loại khác:
```python
sex_le = LabelEncoder()
X['Sex'] = sex_le.fit_transform(X['Sex'])
```
- Sử dụng LabelEncoder để chuyển text thành số
- Mỗi giá trị duy nhất được gán một số nguyên

### Bước 4: Chuẩn hóa dữ liệu số
```python
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
```
- Chuẩn hóa chiều dài và cân nặng về phân phối chuẩn (mean=0, std=1)
- Công thức: `z = (x - mean) / std`
- Giúp model hội tụ nhanh hơn và chính xác hơn

### Bước 5: Chia dữ liệu Train/Test
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
```
- Chia 80% cho training, 20% cho testing
- `stratify=y_encoded`: Đảm bảo tỷ lệ các lớp trong train và test giống nhau
- `random_state=42`: Đảm bảo kết quả có thể tái lập

### Bước 6: Train Model J-48
```python
model = DecisionTreeClassifier(
    criterion='entropy',      # Sử dụng entropy (giống J-48)
    max_depth=None,            # Không giới hạn độ sâu
    min_samples_split=2,       # Tối thiểu 2 mẫu để phân chia
    min_samples_leaf=1,       # Mỗi leaf tối thiểu 1 mẫu
    random_state=42
)
model.fit(X_train, y_train)
```

**Giải thích tham số:**
- `criterion='entropy'`: Sử dụng entropy để đo độ hỗn loạn (giống J-48)
- `max_depth=None`: Không giới hạn độ sâu của cây (có thể điều chỉnh để tránh overfitting)
- `min_samples_split=2`: Cần tối thiểu 2 mẫu để tạo node mới
- `min_samples_leaf=1`: Mỗi leaf node cần tối thiểu 1 mẫu

### Bước 7: Đánh giá Model

#### Tính độ chính xác:
```python
accuracy = accuracy_score(y_test, y_pred)
```
- Tỷ lệ dự đoán đúng trên tổng số mẫu test

#### Classification Report:
- **Precision**: Độ chính xác của dự đoán cho mỗi lớp
- **Recall**: Tỷ lệ tìm được các mẫu thực tế của mỗi lớp
- **F1-Score**: Trung bình điều hòa của Precision và Recall

#### Confusion Matrix:
- Ma trận hiển thị số lượng dự đoán đúng/sai cho mỗi lớp

---

## 📈 Đánh giá kết quả

### Các file kết quả trong folder `result/`:

1. **classification_report.txt**
   - Báo cáo chi tiết về độ chính xác
   - Precision, Recall, F1-Score cho từng lớp
   - Confusion Matrix

2. **confusion_matrix.json**
   - Ma trận nhầm lẫn dạng JSON
   - Dễ đọc và xử lý bằng code

3. **metrics_summary.json**
   - Tổng hợp các metrics chính
   - Accuracy, Precision, Recall, F1-Score

### Cách đọc kết quả:

#### Accuracy (Độ chính xác):
- **> 90%**: Rất tốt
- **80-90%**: Tốt
- **70-80%**: Chấp nhận được
- **< 70%**: Cần cải thiện

#### Precision và Recall:
- **Precision cao**: Ít dự đoán sai dương (false positive)
- **Recall cao**: Tìm được nhiều mẫu thực tế (ít false negative)

#### F1-Score:
- Cân bằng giữa Precision và Recall
- Giá trị càng cao càng tốt (tối đa = 1.0)

---

## 🔍 Troubleshooting

### Lỗi: File không tồn tại
```
FileNotFoundError: crocodile_processed_complete.csv
```
**Giải pháp:** Đảm bảo file CSV có trong cùng thư mục với script

### Lỗi: Thiếu thư viện
```
ModuleNotFoundError: No module named 'pandas'
```
**Giải pháp:** 
```bash
pip install pandas numpy scikit-learn joblib
```

### Lỗi: Encoding trên Windows
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
**Giải pháp:** 
```bash
set PYTHONIOENCODING=utf-8
python train_model.py
```

### Độ chính xác thấp
**Nguyên nhân có thể:**
- Dữ liệu không đủ
- Features không phù hợp
- Model bị overfitting hoặc underfitting

**Giải pháp:**
- Tăng số lượng dữ liệu
- Điều chỉnh `max_depth` để tránh overfitting
- Thử các thuật toán khác (Random Forest, XGBoost)

---

## 📝 Tóm tắt

1. **Chuẩn bị**: Cài đặt Python và các thư viện cần thiết
2. **Dữ liệu**: Đảm bảo file CSV có đầy đủ các cột cần thiết
3. **Train**: Chạy script `train_model.py`
4. **Đánh giá**: Kiểm tra kết quả trong folder `result/`
5. **Sử dụng**: Load model và sử dụng cho dự đoán

---

## 🔗 Tài liệu tham khảo

- [scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [Weka J48 Algorithm](https://weka.sourceforge.io/doc.stable/weka/classifiers/trees/J48.html)
- [Entropy và Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)

---

**Tác giả:** Bùi Hoàn Duy - Nguyễn Tuấn Kiệt - Võ Minh Thắng - Nguyễn Bình Tiến

**Ngày tạo:** 2025




