# config.py - เก็บ API key / config ทั้งหมด

# Supabase Configuration
SUPABASE_URL = "https://rtomduroqzughdtpaiwo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0b21kdXJvcXp1Z2hkdHBhaXdvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQxOTU2NzYsImV4cCI6MjA4OTc3MTY3Nn0.SMsu4cWZPhj7iuGuMV8525M7EfssQ8ePRNVqgTOXrFQ"

# Face Recognition Settings
FACE_RECOGNITION_TOLERANCE = 0.5      # ค่าต่ำ = เข้มงวดมากขึ้น (0.4-0.6)
FACE_ENCODING_MODEL = "large"          # "large" = แม่นกว่า, "small" = เร็วกว่า

# Blink Detection Settings (EAR)
EAR_THRESHOLD = 0.25                   # ค่า EAR ต่ำกว่านี้ = หลับตา
EAR_CONSEC_FRAMES = 2                  # จำนวน frame ติดต่อกันที่ถือว่ากระพริบตา
BLINK_REQUIRED = 2                     # จำนวนกระพริบตาที่ต้องการ (anti-spoofing)
BLINK_TIMEOUT = 10                     # timeout วินาที สำหรับรอกระพริบตา

# Camera Settings
CAMERA_INDEX = 0                       # 0 = กล้องหลัก
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Dataset Path
DATASET_PATH = "dataset"               # โฟลเดอร์เก็บรูป user

# dlib Model Paths
SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"

# Logging
LOG_LEVEL = "INFO"
