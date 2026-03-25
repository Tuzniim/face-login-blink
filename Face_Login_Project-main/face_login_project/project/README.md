# 🔐 Face Recognition Login System with Blink Detection

## โครงสร้างโปรเจค
```
project/
├── dataset/                  # เก็บรูป user (ตอน register)
├── models/                   # dlib model files
├── core/
│   ├── face_recognition_core.py   # detect + compare face
│   ├── blink_detection.py         # EAR + ตรวจจับกระพริบตา
│   └── encoding.py                # แปลง face → vector
├── database/
│   ├── db_connect.py         # connect Supabase
│   ├── user_model.py         # insert / select users
│   └── log_model.py          # insert logs
├── services/
│   ├── register_service.py   # ลงทะเบียน user
│   ├── train_service.py      # encode + save DB
│   └── login_service.py      # login + blink + compare
├── gui/
│   └── app.py                # Tkinter GUI
├── analytics/
│   └── accuracy_plot.py      # วาดกราฟจาก logs
├── config.py                 # เก็บ API key / config
├── requirements.txt
└── main.py                   # entry point
```

## การติดตั้ง

### 1. ติดตั้ง Python dependencies
```bash
pip install -r requirements.txt
```

### 2. ดาวน์โหลด dlib model
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
mkdir -p models
mv shape_predictor_68_face_landmarks.dat models/
```

### 3. ตั้งค่า Supabase
แก้ไข `config.py`:
```python
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

### 4. สร้าง Tables ใน Supabase
```sql
-- Users table
CREATE TABLE users (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  username TEXT UNIQUE NOT NULL,
  full_name TEXT,
  face_encoding JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  is_active BOOLEAN DEFAULT TRUE
);

-- Login logs table
CREATE TABLE login_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  username TEXT,
  status TEXT NOT NULL,
  confidence FLOAT,
  blink_count INTEGER,
  message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 5. รันโปรแกรม
```bash
python main.py
```

## การใช้งาน

### ลงทะเบียน User ใหม่
1. กดปุ่ม **📝 Register**
2. กรอก Username และ Full Name
3. กด **ถ่ายรูปและลงทะเบียน**
4. กด `SPACE` เพื่อถ่ายรูป 5 รูป

### Login
1. กดปุ่ม **🚀 Login**
2. **กระพริบตา 2 ครั้ง** (Anti-Spoofing)
3. รอระบบตรวจจับใบหน้า
4. ดูผลลัพธ์

## Anti-Spoofing (Blink Detection)
ระบบใช้ **Eye Aspect Ratio (EAR)** จาก dlib 68 landmarks
- EAR < 0.25 = ตาหลับ
- ต้องกระพริบตา **2 ครั้ง** ภายใน **10 วินาที**
- ป้องกันการใช้รูปภาพหรือวิดีโอแทนตัวจริง
