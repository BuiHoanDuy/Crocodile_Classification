# 📚 Hướng Dẫn Chi Tiết: Tiền Xử Lý Dữ Liệu và Xây Dựng Mô Hình Dự Đoán Tình Trạng Bảo Tồn Cá Sấu

## 📋 Mục Lục

1. [Tổng quan](#tổng-quan)
2. [Phần 1: Khai báo thư viện và đọc dữ liệu](#phần-1-khai-báo-thư-viện-và-đọc-dữ-liệu)
3. [Phần 2: Làm sạch dữ liệu](#phần-2-làm-sạch-dữ-liệu)
4. [Phần 3: Xử lý Outlier (Capping)](#phần-3-xử-lý-outlier-capping)
5. [Phần 4: Mã hóa và Chuẩn hóa](#phần-4-mã-hóa-và-chuẩn-hóa)
6. [Phần 5: Tích hợp dữ liệu](#phần-5-tích-hợp-dữ-liệu)
7. [Phần 6: Phân tích tương quan](#phần-6-phân-tích-tương-quan)
8. [Phần 7: Chuẩn bị dữ liệu Train/Test](#phần-7-chuẩn-bị-dữ-liệu-traintest)
9. [Phần 8: Trực quan hóa dữ liệu](#phần-8-trực-quan-hóa-dữ-liệu)
10. [Phần 9: Xây dựng và đánh giá mô hình](#phần-9-xây-dựng-và-đánh-giá-mô-hình)
11. [Phần 10: So sánh các mô hình](#phần-10-so-sánh-các-mô-hình)
12. [Phần 11: Demo ứng dụng](#phần-11-demo-ứng-dụng)

---

## 🎯 Tổng quan

Notebook này thực hiện quy trình hoàn chỉnh từ tiền xử lý dữ liệu đến xây dựng và đánh giá các mô hình machine learning để dự đoán tình trạng bảo tồn của cá sấu. Dữ liệu bao gồm các thông tin về chiều dài, cân nặng, nhóm tuổi, giới tính, quốc gia, môi trường sống và khu vực địa lý.

### Mục tiêu:
- Tiền xử lý dữ liệu để loại bỏ nhiễu và chuẩn hóa
- Xây dựng các mô hình phân loại: Decision Tree (J48), Naive Bayes
- Áp dụng K-Means clustering để phân cụm dữ liệu
- So sánh hiệu suất các mô hình
- Demo ứng dụng thực tế

---

## 📦 Phần 1: Khai báo thư viện và đọc dữ liệu

### Code:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('crocodile_dataset.csv')
```

### Giải thích:

**Thư viện được sử dụng:**
- **pandas**: Đọc và xử lý dữ liệu dạng bảng (DataFrame)
- **numpy**: Tính toán số học và xử lý mảng
- **seaborn & matplotlib**: Vẽ biểu đồ trực quan hóa dữ liệu
- **sklearn.preprocessing**: Mã hóa và chuẩn hóa dữ liệu
- **sklearn.model_selection**: Chia dữ liệu train/test

**Đọc dữ liệu:**
- Sử dụng `pd.read_csv()` để đọc file CSV vào DataFrame
- File CSV chứa thông tin về các quan sát cá sấu với các cột như:
  - Observation ID, Common Name, Scientific Name
  - Observed Length (m), Observed Weight (kg)
  - Age Class, Sex, Country/Region, Habitat Type
  - Conservation Status (biến mục tiêu)

**Lưu ý:**
- Sử dụng `try-except` để xử lý lỗi nếu file không tồn tại
- `df.head(3)` hiển thị 3 dòng đầu để kiểm tra dữ liệu

---

## 🧹 Phần 2: Làm sạch dữ liệu

### Code:
```python
# Xử lý giá trị 'Unknown' trong cột Sex
if 'Sex' in df.columns:
    mask = df['Sex'].str.lower().isin(['unknown', 'unkown'])
    random_sex = np.random.choice(['Male', 'Female'], size=mask.sum())
    df.loc[mask, 'Sex'] = random_sex

# Loại bỏ các cột không cần thiết
cols_to_drop = ['Observation ID', 'Observer Name', 'Notes', 
                'Date of Observation', 'Common Name', 
                'Scientific Name', 'Family', 'Genus']
df_clean = df.drop(columns=cols_to_drop, errors='ignore')
```

### Giải thích:

#### 2.1. Xử lý giá trị thiếu trong cột Sex

**Vấn đề:** Một số dòng có giá trị 'Unknown' hoặc 'Unkown' (lỗi chính tả) trong cột giới tính.

**Giải pháp:**
- Tìm tất cả các dòng có giá trị 'unknown' hoặc 'unkown' (không phân biệt hoa thường)
- Random gán ngẫu nhiên 'Male' hoặc 'Female' cho các giá trị này
- Sử dụng `np.random.choice()` để đảm bảo phân phối ngẫu nhiên

**Tại sao làm vậy?**
- Giữ lại dữ liệu thay vì xóa (không mất mẫu)
- Random gán giúp tránh bias trong dữ liệu
- Giới tính là biến quan trọng trong phân loại

#### 2.2. Loại bỏ cột không cần thiết

**Các cột bị loại bỏ:**
- `Observation ID`: ID quan sát (không có giá trị dự đoán)
- `Observer Name`: Tên người quan sát (không liên quan)
- `Notes`: Ghi chú (dữ liệu không cấu trúc)
- `Date of Observation`: Ngày quan sát (không sử dụng)
- `Common Name`, `Scientific Name`, `Family`, `Genus`: Thông tin phân loại sinh học (có thể gây rò rỉ dữ liệu)

**Tại sao loại bỏ?**
- **Tránh rò rỉ dữ liệu (Data Leakage)**: Các cột như Scientific Name có thể chứa thông tin về Conservation Status
- **Giảm độ phức tạp**: Loại bỏ các biến không có giá trị dự đoán
- **Tăng tốc độ xử lý**: Ít cột hơn = tính toán nhanh hơn

**Kết quả:** DataFrame còn lại 7 cột quan trọng:
- Observed Length (m)
- Observed Weight (kg)
- Age Class
- Sex
- Country/Region
- Habitat Type
- Conservation Status

---

## 📊 Phần 3: Xử lý Outlier (Capping)

### Code:
```python
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # Tứ phân vị thứ nhất (25%)
    Q3 = df[column].quantile(0.75)  # Tứ phân vị thứ ba (75%)
    IQR = Q3 - Q1                    # Khoảng tứ phân vị
    lower = Q1 - 1.5 * IQR           # Giới hạn dưới
    upper = Q3 + 1.5 * IQR           # Giới hạn trên
    
    # Capping: Gán giá trị ngoài khoảng về giới hạn
    df[column] = np.where(df[column] < lower, lower,
                          np.where(df[column] > upper, upper, df[column]))
    return df

df_clean = cap_outliers(df_clean, 'Observed Length (m)')
df_clean = cap_outliers(df_clean, 'Observed Weight (kg)')
```

### Giải thích:

#### 3.1. Phương pháp IQR (Interquartile Range)

**IQR là gì?**
- IQR = Q3 - Q1 (khoảng cách giữa tứ phân vị thứ ba và thứ nhất)
- Q1: Giá trị tại vị trí 25% của dữ liệu
- Q3: Giá trị tại vị trí 75% của dữ liệu

**Công thức phát hiện outlier:**
- **Outlier dưới**: Giá trị < Q1 - 1.5 × IQR
- **Outlier trên**: Giá trị > Q3 + 1.5 × IQR

**Ví dụ:**
```
Nếu Q1 = 1.64m, Q3 = 3.01m
IQR = 3.01 - 1.64 = 1.37m
Lower bound = 1.64 - 1.5 × 1.37 = -0.415m
Upper bound = 3.01 + 1.5 × 1.37 = 5.065m
```

#### 3.2. Phương pháp Capping

**Capping là gì?**
- Thay vì xóa outlier, ta "giới hạn" giá trị về biên
- Giá trị < lower bound → gán = lower bound
- Giá trị > upper bound → gán = upper bound

**Ưu điểm của Capping:**
- ✅ Giữ lại tất cả mẫu (không mất dữ liệu)
- ✅ Giảm ảnh hưởng của outlier đến model
- ✅ Phù hợp với dữ liệu có thể có giá trị cực đoan hợp lệ

**So sánh với phương pháp khác:**
- **Xóa outlier**: Mất dữ liệu, có thể làm giảm số lượng mẫu
- **Winsorization**: Tương tự capping nhưng có thể giữ nhiều outlier hơn
- **Z-score**: Dựa trên độ lệch chuẩn, nhạy cảm với phân phối không chuẩn

#### 3.3. Trực quan hóa với Boxplot

```python
sns.boxplot(y=df_clean['Observed Length (m)'])
```

**Boxplot hiển thị:**
- Q1, Q2 (median), Q3
- Whiskers: Giới hạn trên/dưới (thường là Q1-1.5×IQR và Q3+1.5×IQR)
- Outliers: Các điểm nằm ngoài whiskers

---

## 🔢 Phần 4: Mã hóa và Chuẩn hóa

### Code:
```python
# Mã hóa biến mục tiêu
target_le = LabelEncoder()
df_clean['Conservation Status'] = target_le.fit_transform(df_clean['Conservation Status'])

# Mã hóa Age Class
age_mapping = {'Hatchling': 0, 'Juvenile': 1, 'Subadult': 2, 'Adult': 3}
df_clean['Age Class'] = df_clean['Age Class'].map(age_mapping)

# Mã hóa các biến phân loại khác
cat_cols = ['Sex', 'Country/Region', 'Habitat Type']
le = LabelEncoder()
for col in cat_cols:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
num_cols = ['Observed Length (m)', 'Observed Weight (kg)']
df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
```

### Giải thích:

#### 4.1. Mã hóa biến mục tiêu (Target Encoding)

**LabelEncoder:**
- Chuyển đổi các nhãn text thành số nguyên
- Ví dụ: 'Critically Endangered' → 0, 'Data Deficient' → 1, ...

**Mapping nhãn:**
```
'Critically Endangered': 0
'Data Deficient': 1
'Endangered': 2
'Least Concern': 3
'Vulnerable': 4
```

**Lưu ý:** 
- Cần lưu mapping để giải mã lại sau khi dự đoán
- Thứ tự mã hóa quan trọng cho một số thuật toán

#### 4.2. Mã hóa Age Class

**Ordinal Encoding:**
- Sử dụng mapping thủ công vì Age Class có thứ tự tự nhiên
- Hatchling (0) < Juvenile (1) < Subadult (2) < Adult (3)

**Tại sao không dùng LabelEncoder?**
- LabelEncoder không đảm bảo thứ tự
- Mapping thủ công giữ được ý nghĩa thứ tự của dữ liệu

#### 4.3. Mã hóa các biến phân loại khác

**LabelEncoder cho Sex, Country/Region, Habitat Type:**
- Mỗi giá trị duy nhất được gán một số nguyên
- Ví dụ: 'Male' → 0, 'Female' → 1
- 'Vietnam' → 0, 'Thailand' → 1, ...

**Lưu ý:**
- Cần fit riêng cho mỗi cột (không dùng chung encoder)
- Sử dụng `astype(str)` để đảm bảo xử lý đúng các giá trị đặc biệt

#### 4.4. Chuẩn hóa dữ liệu số (Standardization)

**StandardScaler:**
- Chuyển đổi dữ liệu về phân phối chuẩn với mean=0, std=1
- Công thức: `z = (x - mean) / std`

**Ví dụ:**
```
Trước chuẩn hóa:
- Mean Length = 2.42m, Std = 1.10m
- Mean Weight = 155.77kg, Std = 175.19kg

Sau chuẩn hóa:
- Mean Length = 0, Std = 1
- Mean Weight = 0, Std = 1
```

**Tại sao cần chuẩn hóa?**
- ✅ Các thuật toán dựa trên khoảng cách (K-Means, SVM) hoạt động tốt hơn
- ✅ Gradient descent hội tụ nhanh hơn
- ✅ Tránh bias do scale khác nhau giữa các biến
- ✅ Một số thuật toán yêu cầu dữ liệu đã chuẩn hóa

**So sánh với Normalization (Min-Max Scaling):**
- **StandardScaler**: Mean=0, Std=1 (phù hợp khi dữ liệu có phân phối chuẩn)
- **MinMaxScaler**: Scale về [0, 1] (phù hợp khi cần giữ nguyên phân phối)

---

## 🌍 Phần 5: Tích hợp dữ liệu

### Code:
```python
country_to_continent = {
    'Australia': 'Oceania',
    'Vietnam': 'Southeast Asia',
    'India': 'South Asia',
    # ... mapping đầy đủ
}

def get_continent(country):
    return country_to_continent.get(country, 'Other')

df_clean['Continent'] = df_clean['Country/Region'].apply(get_continent)
```

### Giải thích:

#### 5.1. Tạo cột Continent từ Country/Region

**Mục đích:**
- Tạo feature mới từ feature hiện có (Feature Engineering)
- Giảm số lượng giá trị duy nhất (47 quốc gia → 13 khu vực)
- Giúp model học được pattern theo khu vực địa lý

**Mapping các khu vực:**
- **Oceania**: Australia, Papua New Guinea
- **Southeast Asia**: Vietnam, Thailand, Cambodia, ...
- **South Asia**: India, Sri Lanka, Pakistan, Nepal
- **West Africa**: Ghana, Nigeria, Liberia, ...
- **Central Africa**: Cameroon, Congo (DRC), ...
- **East Africa**: Kenya, Uganda, Tanzania, ...
- **Northern Africa**: Egypt
- **North America**: USA (Florida), Mexico
- **Central America**: Costa Rica, Guatemala, Belize
- **Caribbean**: Cuba
- **South America**: Colombia, Venezuela
- **Western Asia**: Iran (historic)
- **Southern Africa**: South Africa
- **Other**: Các quốc gia không có trong mapping

#### 5.2. Phân bố mẫu theo khu vực

**Kết quả phân bố:**
```
Southeast Asia     229 mẫu (22.9%)
Oceania            151 mẫu (15.1%)
West Africa        147 mẫu (14.7%)
Central Africa     121 mẫu (12.1%)
South America       79 mẫu (7.9%)
Caribbean           77 mẫu (7.7%)
Central America     48 mẫu (4.8%)
South Asia          46 mẫu (4.6%)
North America       43 mẫu (4.3%)
East Africa         37 mẫu (3.7%)
Western Asia        11 mẫu (1.1%)
Southern Africa      8 mẫu (0.8%)
Northern Africa      3 mẫu (0.3%)
```

**Nhận xét:**
- Dữ liệu không cân bằng giữa các khu vực
- Southeast Asia và Oceania chiếm tỷ lệ cao nhất
- Một số khu vực có rất ít mẫu (Northern Africa chỉ có 3 mẫu)

**Ảnh hưởng:**
- Model có thể bias về các khu vực có nhiều dữ liệu
- Cần cân nhắc khi dự đoán cho các khu vực ít dữ liệu

---

## 📈 Phần 6: Phân tích tương quan

### Code:
```python
correlation_matrix = df_clean[['Observed Length (m)', 'Observed Weight (kg)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
```

### Giải thích:

#### 6.1. Ma trận tương quan Pearson

**Hệ số tương quan:**
- Đo mức độ quan hệ tuyến tính giữa hai biến
- Giá trị từ -1 đến +1:
  - **+1**: Tương quan dương hoàn hảo
  - **0**: Không có tương quan
  - **-1**: Tương quan âm hoàn hảo

**Kết quả:**
```
Chiều dài vs Cân nặng: r = 0.8434
```

**Giải thích:**
- Hệ số tương quan **0.8434** cho thấy có tương quan dương mạnh
- Cá sấu dài hơn thường nặng hơn (điều này hợp lý về mặt sinh học)
- Tương quan cao có thể gây đa cộng tuyến (multicollinearity)

#### 6.2. Heatmap trực quan hóa

**Heatmap hiển thị:**
- Màu đỏ: Tương quan dương
- Màu xanh: Tương quan âm
- Màu trắng: Không tương quan

**Ứng dụng:**
- Phát hiện các biến có tương quan cao (có thể loại bỏ một trong hai)
- Hiểu mối quan hệ giữa các biến

**Lưu ý:**
- Trong trường hợp này, giữ cả hai biến vì:
  - Cả hai đều có giá trị dự đoán
  - Một số thuật toán có thể xử lý được đa cộng tuyến

---

## 🎯 Phần 7: Chuẩn bị dữ liệu Train/Test

### Code:
```python
# Mã hóa lại các biến
le_target = LabelEncoder()
df_clean['Conservation Status'] = le_target.fit_transform(df_clean['Conservation Status'])

age_mapping = {'Hatchling': 0, 'Juvenile': 1, 'Subadult': 2, 'Adult': 3}
df_clean['Age Class'] = df_clean['Age Class'].map(age_mapping)

cat_cols_to_encode = ['Sex', 'Country/Region', 'Habitat Type', 'Continent']
le_features = LabelEncoder()
for col in cat_cols_to_encode:
    df_clean[col] = le_features.fit_transform(df_clean[col])

# Chuẩn hóa lại
scaler = StandardScaler()
num_cols = ['Observed Length (m)', 'Observed Weight (kg)']
df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

# Chia dữ liệu
X = df_clean.drop('Conservation Status', axis=1)
y = df_clean['Conservation Status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```

### Giải thích:

#### 7.1. Mã hóa lại dữ liệu

**Tại sao mã hóa lại?**
- Đảm bảo tất cả biến đã được mã hóa đúng cách
- Chuẩn bị cho việc train model
- Lưu các encoder để sử dụng sau này

**Lưu ý:**
- Cần fit encoder trên toàn bộ dữ liệu trước khi chia train/test
- Nếu fit riêng trên train/test, có thể gây mismatch

#### 7.2. Chia dữ liệu Train/Test

**Tỷ lệ chia:**
- **Train: 70%** (700 mẫu)
- **Test: 30%** (300 mẫu)

**Stratified Split:**
- `stratify=y`: Đảm bảo tỷ lệ các lớp trong train và test giống nhau
- Tránh trường hợp một lớp chỉ có trong train hoặc test

**Ví dụ phân bố:**
```
Train Set:
- Critically Endangered: 27.57%
- Data Deficient: 11.43%
- Endangered: 5.57%
- Least Concern: 38.43%
- Vulnerable: 17.00%

Test Set:
- Critically Endangered: 27.33%
- Data Deficient: 11.67%
- Endangered: 5.67%
- Least Concern: 38.33%
- Vulnerable: 17.00%
```

**random_state=42:**
- Đảm bảo kết quả có thể tái lập
- Cùng seed sẽ cho cùng kết quả chia dữ liệu

---

## 📊 Phần 8: Trực quan hóa dữ liệu

### Code:
```python
# Phân bố biến mục tiêu
sns.countplot(x=y, hue=y, palette='viridis')

# Scatter plot chiều dài vs cân nặng
sns.scatterplot(data=df_clean, x='Observed Length (m)', 
                y='Observed Weight (kg)', hue='Age Class')

# Thống kê mô tả
df_clean[['Observed Length (m)', 'Observed Weight (kg)']].describe()
```

### Giải thích:

#### 8.1. Phân bố biến mục tiêu

**Countplot:**
- Hiển thị số lượng mẫu cho mỗi lớp
- Giúp phát hiện class imbalance

**Kết quả:**
- Least Concern: 384 mẫu (38.4%) - Lớp đa số
- Critically Endangered: 275 mẫu (27.5%)
- Vulnerable: 170 mẫu (17.0%)
- Data Deficient: 115 mẫu (11.5%)
- Endangered: 56 mẫu (5.6%) - Lớp thiểu số

**Vấn đề Class Imbalance:**
- Model có thể bias về lớp đa số
- Cần cân nhắc sử dụng:
  - Class weights
  - SMOTE (oversampling)
  - Undersampling

#### 8.2. Scatter Plot

**Mục đích:**
- Trực quan hóa mối quan hệ giữa chiều dài và cân nặng
- Phân biệt theo nhóm tuổi (Age Class)

**Nhận xét:**
- Có xu hướng tuyến tính giữa chiều dài và cân nặng
- Các nhóm tuổi khác nhau có phân bố khác nhau
- Adult thường có chiều dài và cân nặng lớn hơn

#### 8.3. Thống kê mô tả

**Các chỉ số:**
- **count**: Số lượng mẫu
- **mean**: Giá trị trung bình
- **std**: Độ lệch chuẩn
- **min, 25%, 50% (median), 75%, max**: Các tứ phân vị

**Sau chuẩn hóa:**
- Mean ≈ 0, Std ≈ 1 (đúng như mong đợi)

---

## 🤖 Phần 9: Xây dựng và đánh giá mô hình

### 9.1. Decision Tree (J48)

#### Code:
```python
j48_model = DecisionTreeClassifier(
    criterion='entropy',      # Sử dụng entropy (giống J48)
    max_depth=4,              # Giới hạn độ sâu
    random_state=42
)
j48_model.fit(X_train, y_train)
```

#### Giải thích:

**Decision Tree với Entropy:**
- **criterion='entropy'**: Sử dụng Information Gain để chọn feature tốt nhất
- **max_depth=4**: Giới hạn độ sâu để tránh overfitting

**Kết quả:**
- Accuracy trên Test Set: **72.67%**
- Precision, Recall, F1-Score khác nhau cho từng lớp

**Phân tích:**
- Critically Endangered: Recall = 1.00 (tìm được tất cả), nhưng Precision = 0.52 (nhiều dự đoán sai)
- Least Concern: Precision = 0.97, Recall = 0.68 (dự đoán đúng nhưng bỏ sót một số)
- Data Deficient: Precision = 0.95 nhưng Recall = 0.51 (dự đoán đúng nhưng bỏ sót nhiều)

**Visualization:**
- Vẽ cây quyết định để hiểu logic của model
- Export rules dạng text để giải thích

### 9.2. Decision Tree với max_depth=5

#### Code:
```python
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)
```

#### Kết quả:
- **Train Accuracy: 79.57%**
- **Test Accuracy: 80.00%**

**So sánh với max_depth=4:**
- Tăng độ sâu → tăng độ chính xác
- Gap giữa train và test nhỏ → không bị overfitting nhiều

### 9.3. K-Means Clustering

#### Code:
```python
kmeans = KMeans(n_clusters=5, random_state=42)  # 5 clusters = 5 classes
kmeans.fit(X_train)
```

#### Giải thích:

**K-Means:**
- Phân cụm dữ liệu thành 5 nhóm (tương ứng với 5 lớp)
- Không phải thuật toán phân loại, nhưng có thể dùng để phân cụm

**Silhouette Score:**
- Đo chất lượng phân cụm
- Giá trị từ -1 đến +1:
  - **+1**: Phân cụm tốt
  - **0**: Chồng chéo giữa các cụm
  - **-1**: Phân cụm sai

**Kết quả:**
- Silhouette Score (Train): **0.4019**
- Silhouette Score (Test): **0.4084**

**Nhận xét:**
- Score trung bình (~0.40) cho thấy phân cụm chấp nhận được
- Không tốt bằng Decision Tree cho bài toán phân loại

### 9.4. Naive Bayes

#### Code:
```python
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
```

#### Giải thích:

**Gaussian Naive Bayes:**
- Giả định các features độc lập với nhau (naive assumption)
- Sử dụng phân phối chuẩn (Gaussian) cho các biến số

**Kết quả:**
- **Train Accuracy: 38.71%**
- **Test Accuracy: 35.33%**

**Phân tích:**
- Hiệu suất thấp nhất trong các mô hình
- Có thể do:
  - Giả định độc lập không phù hợp (các biến có tương quan)
  - Phân phối không chuẩn sau khi chuẩn hóa
  - Dữ liệu không phù hợp với giả định của Naive Bayes

---

## 📊 Phần 10: So sánh các mô hình

### Code:
```python
models = ['Decision Tree (J48)', 'Naive Bayes']
accuracies = [80.00, 35.33]  # Test accuracy
plt.bar(models, accuracies)
```

### Kết quả so sánh:

| Mô hình | Train Accuracy | Test Accuracy | Nhận xét |
|---------|---------------|---------------|----------|
| **Decision Tree (J48)** | 79.57% | 80.00% | ✅ Tốt nhất, không overfitting |
| **Naive Bayes** | 38.71% | 35.33% | ❌ Hiệu suất thấp |
| **K-Means** | - | Silhouette: 40.84% | ⚠️ Không phù hợp cho phân loại |

### Kết luận:

1. **Decision Tree (J48) là lựa chọn tốt nhất:**
   - Độ chính xác cao (80%)
   - Không bị overfitting (train ≈ test)
   - Dễ giải thích (cây quyết định)

2. **Naive Bayes không phù hợp:**
   - Giả định độc lập không đúng với dữ liệu
   - Cần dữ liệu phù hợp hơn với giả định

3. **K-Means:**
   - Phù hợp cho clustering, không phải classification
   - Có thể dùng để phân cụm dữ liệu trước khi phân loại

---

## 🎮 Phần 11: Demo ứng dụng

### Code:
```python
def predict_crocodile_status(length, weight, habitat_code):
    # Tạo input từ thông tin đầu vào
    # Chuẩn hóa dữ liệu
    # Dự đoán bằng model
    return prediction

# Kịch bản 1: Cá sấu con
result1 = predict_crocodile_status(0.8, 5.0, 1)
# → "CRITICALLY ENDANGERED - CẦN BẢO VỆ!"

# Kịch bản 2: Cá sấu trưởng thành
result2 = predict_crocodile_status(4.5, 300.0, 2)
# → "Least Concern"
```

### Giải thích:

#### 11.1. Hàm dự đoán

**Input:**
- `length`: Chiều dài (m)
- `weight`: Cân nặng (kg)
- `habitat_code`: Mã môi trường sống

**Xử lý:**
1. Tạo DataFrame từ input
2. Chuẩn hóa dữ liệu số (sử dụng scaler đã fit)
3. Mã hóa các biến phân loại
4. Dự đoán bằng model
5. Giải mã nhãn về tên gốc

**Output:**
- Tình trạng bảo tồn dự đoán

#### 11.2. Kịch bản ứng dụng

**Kịch bản 1: Cá sấu con**
- Chiều dài: 0.8m (rất nhỏ)
- Cân nặng: 5kg
- → Dự đoán: **Critically Endangered**
- → Hành động: Cần bảo vệ ngay lập tức

**Kịch bản 2: Cá sấu trưởng thành**
- Chiều dài: 4.5m (lớn)
- Cân nặng: 300kg
- → Dự đoán: **Least Concern**
- → Hành động: Tình trạng ổn định

#### 11.3. Ứng dụng thực tế

**Các ứng dụng có thể:**
1. **Hệ thống giám sát tự động:**
   - Camera tự động đo kích thước
   - Hệ thống cảnh báo khi phát hiện cá sấu nguy cấp

2. **Ứng dụng di động:**
   - Kiểm lâm nhập thông tin quan sát
   - Nhận cảnh báo và khuyến nghị

3. **Phân tích dữ liệu lớn:**
   - Phân tích xu hướng bảo tồn
   - Dự đoán tình trạng trong tương lai

---

## 📝 Tổng kết

### Quy trình đã thực hiện:

1. ✅ **Đọc và kiểm tra dữ liệu**
2. ✅ **Làm sạch dữ liệu** (xử lý missing values, loại bỏ cột không cần)
3. ✅ **Xử lý outlier** (IQR capping)
4. ✅ **Mã hóa và chuẩn hóa** (LabelEncoder, StandardScaler)
5. ✅ **Feature Engineering** (tạo cột Continent)
6. ✅ **Phân tích tương quan**
7. ✅ **Chia dữ liệu train/test** (stratified split)
8. ✅ **Trực quan hóa dữ liệu**
9. ✅ **Xây dựng mô hình** (Decision Tree, Naive Bayes, K-Means)
10. ✅ **Đánh giá và so sánh mô hình**
11. ✅ **Demo ứng dụng**

### Kết quả:

- **Mô hình tốt nhất**: Decision Tree (J48) với độ chính xác **80%**
- **Dữ liệu sau xử lý**: Sạch, chuẩn hóa, sẵn sàng cho machine learning
- **Ứng dụng**: Có thể tích hợp vào hệ thống thực tế

### Lưu ý:

- Dữ liệu có class imbalance → cân nhắc sử dụng class weights
- Một số khu vực có ít dữ liệu → cần thêm dữ liệu hoặc xử lý đặc biệt
- Model có thể cải thiện bằng cách:
  - Tăng số lượng dữ liệu
  - Feature engineering tốt hơn
  - Thử các thuật toán khác (Random Forest, XGBoost)

---

**Tác giả:** Bùi Hoàn Duy - Nguyễn Tuấn Kiệt - Võ Minh Thắng - Nguyễn Bình Tiến

**Ngày tạo:** 2025


