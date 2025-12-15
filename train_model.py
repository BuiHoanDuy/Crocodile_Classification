"""
Script train model J-48 (Decision Tree) cho du doan Conservation Status cua ca sau
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import json
import sys
import os

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding
        os.system('chcp 65001 >nul 2>&1')
        # Also set environment variable
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    except:
        pass

print("=" * 60)
print("TRAIN MODEL J-48 (DECISION TREE)")
print("=" * 60)

# 1. ĐỌC DỮ LIỆU
print("\n[1] Đang đọc dữ liệu...")
df = pd.read_csv('crocodile_processed_complete.csv')
print(f"   ✓ Đã đọc {len(df)} dòng dữ liệu")
print(f"   ✓ Các cột: {list(df.columns)}")

# 2. CHUẨN BỊ DỮ LIỆU
print("\n[2] Đang chuẩn bị dữ liệu...")

# Xử lý giá trị thiếu
df = df.dropna()

# Xác định features và target
feature_cols = [
    'Observed Length (m)',
    'Observed Weight (kg)',
    'Age Class',
    'Sex',
    'Country/Region',
    'Habitat Type',
    'Continent'
]

target_col = 'Conservation Status'

# Kiểm tra các cột có tồn tại
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"   ⚠ Cảnh báo: Thiếu các cột {missing_cols}")
    feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"   ✓ Features: {len(feature_cols)} cột")
print(f"   ✓ Target: {target_col}")
print(f"   ✓ Số lượng mẫu: {len(X)}")

# 3. MÃ HÓA DỮ LIỆU
print("\n[3] Đang mã hóa dữ liệu...")

# Mã hóa Age Class
age_mapping = {'Hatchling': 0, 'Juvenile': 1, 'Subadult': 2, 'Adult': 3}
if 'Age Class' in X.columns:
    X['Age Class'] = X['Age Class'].map(age_mapping).fillna(2)

# Mã hóa Sex
sex_le = LabelEncoder()
if 'Sex' in X.columns:
    X['Sex'] = sex_le.fit_transform(X['Sex'].astype(str))

# Mã hóa Country/Region
country_le = LabelEncoder()
if 'Country/Region' in X.columns:
    X['Country/Region'] = country_le.fit_transform(X['Country/Region'].astype(str))

# Mã hóa Habitat Type
habitat_le = LabelEncoder()
if 'Habitat Type' in X.columns:
    X['Habitat Type'] = habitat_le.fit_transform(X['Habitat Type'].astype(str))

# Mã hóa Continent
continent_le = LabelEncoder()
if 'Continent' in X.columns:
    X['Continent'] = continent_le.fit_transform(X['Continent'].astype(str))

# Mã hóa Target (Conservation Status)
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

print(f"   ✓ Đã mã hóa các biến phân loại")
print(f"   ✓ Số lớp target: {len(target_le.classes_)}")
print(f"   ✓ Các lớp: {list(target_le.classes_)}")

# 4. CHUẨN HÓA DỮ LIỆU SỐ
print("\n[4] Đang chuẩn hóa dữ liệu số...")
numeric_cols = ['Observed Length (m)', 'Observed Weight (kg)']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print(f"   ✓ Đã chuẩn hóa các cột số: {numeric_cols}")

# 5. CHIA DỮ LIỆU TRAIN/TEST
print("\n[5] Đang chia dữ liệu train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   ✓ Train: {len(X_train)} mẫu")
print(f"   ✓ Test: {len(X_test)} mẫu")

# 6. TRAIN MODEL J-48 (DECISION TREE)
print("\n[6] Đang train model J-48...")
# J-48 tương đương với Decision Tree với các tham số:
# - criterion='entropy' (giống J-48)
# - min_samples_split=2 (mặc định)
# - min_samples_leaf=1 (mặc định)
model = DecisionTreeClassifier(
    criterion='entropy',  # J-48 sử dụng entropy
    max_depth=None,       # Không giới hạn độ sâu (có thể điều chỉnh)
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)
print("   ✓ Model đã được train thành công")

# 7. ĐÁNH GIÁ MODEL
print("\n[7] Đang đánh giá model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Tính toán confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Tính toán các metrics chi tiết
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n   📊 KẾT QUẢ ĐÁNH GIÁ:")
print(f"   ✓ Độ chính xác (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   ✓ Precision (weighted): {precision:.4f}")
print(f"   ✓ Recall (weighted): {recall:.4f}")
print(f"   ✓ F1-Score (weighted): {f1:.4f}")
print(f"\n   📋 Classification Report:")
report_str = classification_report(y_test, y_pred, target_names=target_le.classes_)
print(report_str)

# 7.1. LƯU KẾT QUẢ VÀO FOLDER RESULT
print("\n[7.1] Đang lưu kết quả đánh giá vào folder result...")
os.makedirs('result', exist_ok=True)

# Lưu classification report
with open('result/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Model: J-48 (Decision Tree)\n")
    f.write(f"Criterion: entropy\n")
    f.write(f"Train samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    f.write("=" * 60 + "\n")
    f.write("METRICS SUMMARY\n")
    f.write("=" * 60 + "\n")
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision (weighted): {precision:.4f}\n")
    f.write(f"Recall (weighted): {recall:.4f}\n")
    f.write(f"F1-Score (weighted): {f1:.4f}\n\n")
    f.write("=" * 60 + "\n")
    f.write("DETAILED CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(report_str)
    f.write("\n" + "=" * 60 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("=" * 60 + "\n\n")
    f.write("Classes: " + ", ".join(target_le.classes_) + "\n\n")
    f.write(str(cm))
    f.write("\n")

# Lưu confusion matrix dạng JSON
cm_dict = {
    'confusion_matrix': cm.tolist(),
    'classes': target_le.classes_.tolist(),
    'accuracy': float(accuracy),
    'precision_weighted': float(precision),
    'recall_weighted': float(recall),
    'f1_weighted': float(f1)
}

with open('result/confusion_matrix.json', 'w', encoding='utf-8') as f:
    json.dump(cm_dict, f, indent=2, ensure_ascii=False)

# Lưu metrics tổng hợp
metrics_summary = {
    'model_type': 'J-48 (Decision Tree)',
    'criterion': 'entropy',
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'accuracy': float(accuracy),
    'accuracy_percent': float(accuracy * 100),
    'precision_weighted': float(precision),
    'recall_weighted': float(recall),
    'f1_weighted': float(f1),
    'target_classes': target_le.classes_.tolist(),
    'feature_columns': feature_cols
}

with open('result/metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

print("   ✓ Đã lưu classification_report.txt")
print("   ✓ Đã lưu confusion_matrix.json")
print("   ✓ Đã lưu metrics_summary.json")

# 8. LƯU MODEL VÀ CÁC ENCODER
print("\n[8] Đang lưu model và các encoder...")

# Lưu model
joblib.dump(model, 'model_j48.pkl')
print("   ✓ Đã lưu model: model_j48.pkl")

# Lưu các encoder và scaler
joblib.dump(sex_le, 'encoders/sex_encoder.pkl')
joblib.dump(country_le, 'encoders/country_encoder.pkl')
joblib.dump(habitat_le, 'encoders/habitat_encoder.pkl')
joblib.dump(continent_le, 'encoders/continent_encoder.pkl')
joblib.dump(target_le, 'encoders/target_encoder.pkl')
joblib.dump(scaler, 'encoders/scaler.pkl')
print("   ✓ Đã lưu các encoder và scaler")

# Lưu metadata
metadata = {
    'feature_columns': feature_cols,
    'target_classes': target_le.classes_.tolist(),
    'age_mapping': age_mapping,
    'accuracy': float(accuracy),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'model_type': 'J-48 (Decision Tree)',
    'criterion': 'entropy'
}

with open('model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("   ✓ Đã lưu metadata: model_metadata.json")

# Lưu mapping cho API
mapping_data = {
    'sex_classes': sex_le.classes_.tolist() if hasattr(sex_le, 'classes_') else [],
    'country_classes': country_le.classes_.tolist() if hasattr(country_le, 'classes_') else [],
    'habitat_classes': habitat_le.classes_.tolist() if hasattr(habitat_le, 'classes_') else [],
    'continent_classes': continent_le.classes_.tolist() if hasattr(continent_le, 'classes_') else [],
    'target_classes': target_le.classes_.tolist() if hasattr(target_le, 'classes_') else [],
    'age_mapping': age_mapping
}

with open('mappings.json', 'w', encoding='utf-8') as f:
    json.dump(mapping_data, f, indent=2, ensure_ascii=False)
print("   ✓ Đã lưu mappings: mappings.json")

print("\n" + "=" * 60)
print("✅ HOÀN THÀNH TRAIN MODEL!")
print("=" * 60)
print("\nCác file đã tạo:")
print("  - model_j48.pkl (model đã train)")
print("  - encoders/ (thư mục chứa các encoder)")
print("  - model_metadata.json (thông tin model)")
print("  - mappings.json (mapping các giá trị)")
print("  - result/classification_report.txt (báo cáo chi tiết)")
print("  - result/confusion_matrix.json (ma trận nhầm lẫn)")
print("  - result/metrics_summary.json (tổng hợp metrics)")
print("\nTiếp theo: Chạy API với lệnh: uvicorn main:app --reload")

