"""FastAPI 엔트리포인트.

서버 환경에서 matplotlib을 안전하게 사용하기 위해 *반드시* 다른 import보다
먼저 backend Agg를 지정한다.
"""

import matplotlib  # noqa: E402
matplotlib.use('Agg')

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from app.settings import CORS_ORIGINS, FRONTEND_DIST_DIR
from app.api import data as data_router
from app.api import preprocess as preprocess_router
from app.api import train as train_router
from app.api import predict as predict_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Amazon Sales Prediction API',
    version='0.1.0',
    description='판매량 예측 모델의 데이터 업로드/학습/예측 API.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


app.include_router(data_router.router)
app.include_router(preprocess_router.router)
app.include_router(train_router.router)
app.include_router(predict_router.router)
app.include_router(predict_router.models_router)


@app.get('/api/health', tags=['system'])
def health() -> dict:
    """헬스체크 — 토치 버전과 GPU 가용 여부를 반환."""
    import torch
    return {
        'status': 'ok',
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


# ---------- 운영 모드: 빌드된 프론트엔드 정적 서빙 ----------
# 모든 API 라우터 등록 *이후*에 마운트해야 라우트 우선순위가 보장된다.
class SPAStaticFiles(StaticFiles):
    """SPA fallback을 지원하는 StaticFiles.

    /assets, /favicon 등 실제 파일은 그대로 서빙하고, 그 외 경로(deep link 새로고침 등)는
    index.html로 fallback해서 vue-router의 history 모드가 동작하게 한다.

    Starlette는 404를 예외가 아닌 Response로 반환하므로 status_code로 분기한다.
    """

    async def get_response(self, path: str, scope: Scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            # /api/* 가 여기로 빠지면 정상적인 404 JSON을 유지해야 함
            if path.startswith('api/'):
                raise
            return FileResponse(Path(str(self.directory)) / 'index.html')


if FRONTEND_DIST_DIR.exists():
    app.mount('/', SPAStaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name='frontend')
    logger.info('운영 모드 활성: %s 를 정적 서빙', FRONTEND_DIST_DIR)
else:
    logger.info('frontend/dist 미존재 — 개발 모드 (Vite dev server 사용)')
