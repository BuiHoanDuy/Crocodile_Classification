# 🚀 Hướng dẫn nhanh

## Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## Bước 2: Train model
```bash
python train_model.py
```

## Bước 3: Chạy API
```bash
python -m uvicorn main:app --reload
```

## Bước 4: Mở trình duyệt
Truy cập: `http://localhost:8000`

## 📝 Lưu ý
- Đảm bảo file `crocodile_processed_complete.csv` có trong thư mục
- Sau khi train, các file model sẽ được tạo tự động
- API sẽ tự động load model khi khởi động

