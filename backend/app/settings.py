"""앱 설정 / 경로 상수.

모든 파일 I/O는 여기서 정의한 절대 경로를 사용한다 (cwd 의존 제거).
환경 변수로 오버라이드 가능:
  AMAZON_BACKEND_DIR   백엔드 루트 (기본: backend/)
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get(
    'AMAZON_BACKEND_DIR',
    Path(__file__).resolve().parent.parent,
)).resolve()

# 입력 데이터
DATA_PATH = BASE_DIR / 'Amazon Sale Report.csv'

# 전처리 산출물
PREPROC_PATH = BASE_DIR / 'preprocessor.pkl'
NPY_DIR = BASE_DIR
X_TRAIN_PATH = BASE_DIR / 'X_train.npy'
X_VAL_PATH = BASE_DIR / 'X_val.npy'
X_TEST_PATH = BASE_DIR / 'X_test.npy'
Y_TRAIN_PATH = BASE_DIR / 'y_train.npy'
Y_VAL_PATH = BASE_DIR / 'y_val.npy'
Y_TEST_PATH = BASE_DIR / 'y_test.npy'

# 학습/평가 산출물
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
UPLOADS_DIR = BASE_DIR / 'uploads'  # 사용자가 업로드한 CSV 임시 보관

for d in (MODELS_DIR, RESULTS_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# CORS — 개발 시 Vite dev server
CORS_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
]

# 업로드 제한
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# 운영 빌드 정적 서빙
# 프로젝트 루트 = backend/ 의 부모
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST_DIR = PROJECT_ROOT / 'frontend' / 'dist'
