"""전처리 라우터 — `data_preprocessing.main()`을 API로 노출."""

from __future__ import annotations
import logging
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas import PreprocessResponse
from app.settings import (
    DATA_PATH,
    PREPROC_PATH,
    X_TRAIN_PATH, X_VAL_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_VAL_PATH, Y_TEST_PATH,
)

# backend/ 를 path에 넣어 기존 모듈 사용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data_preprocessing import DataPreprocessor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/preprocess', tags=['preprocess'])


@router.post('', response_model=PreprocessResponse)
async def run_preprocess() -> PreprocessResponse:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail='활성 데이터셋이 없습니다. 먼저 업로드하세요.')

    pp = DataPreprocessor()
    target_column = 'Amount'
    date_columns = ['Date']

    df = pp.load_data(str(DATA_PATH))
    df = pp.drop_unwanted_columns(df)
    df = pp.drop_target_missing(df, target_column)
    df = pp.filter_status(df)
    df = pp.transform_promotion_ids(df)
    df = pp.extract_date_features(df, date_columns)

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(np.float32)
    pp.feature_names = X.columns.tolist()
    pp.target_column = target_column

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=42)

    pp.fit_missing(X_train)
    X_train = pp.transform_missing(X_train.copy())
    X_val = pp.transform_missing(X_val.copy())
    X_test = pp.transform_missing(X_test.copy())

    pp.fit_categorical(X_train)
    X_train = pp.transform_categorical(X_train)
    X_val = pp.transform_categorical(X_val)
    X_test = pp.transform_categorical(X_test)

    pp.fit_scaler(X_train)
    X_train = pp.transform_scaler(X_train)
    X_val = pp.transform_scaler(X_val)
    X_test = pp.transform_scaler(X_test)

    np.save(X_TRAIN_PATH, X_train.values.astype(np.float32))
    np.save(X_VAL_PATH, X_val.values.astype(np.float32))
    np.save(X_TEST_PATH, X_test.values.astype(np.float32))
    np.save(Y_TRAIN_PATH, y_train.values.astype(np.float32))
    np.save(Y_VAL_PATH, y_val.values.astype(np.float32))
    np.save(Y_TEST_PATH, y_test.values.astype(np.float32))
    pp.save_preprocessor(str(PREPROC_PATH))

    return PreprocessResponse(
        train=len(X_train),
        val=len(X_val),
        test=len(X_test),
        n_features=len(pp.feature_names),
        feature_names=pp.feature_names,
    )
