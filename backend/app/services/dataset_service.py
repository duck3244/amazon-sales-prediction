"""데이터셋 업로드/요약 서비스."""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from app.settings import DATA_PATH, MAX_UPLOAD_BYTES, UPLOADS_DIR
from app.schemas import ColumnSummary, DatasetSummary


TARGET_COLUMN = 'Amount'


class UploadTooLarge(Exception):
    pass


def save_uploaded_csv(stream: BinaryIO, filename: str,
                      activate: bool = True) -> tuple[Path, int]:
    """업로드 스트림을 디스크에 그대로 저장. 메모리 보호 위해 chunk 단위로 복사.

    Args:
        activate: True면 ``DATA_PATH``(학습용 활성 데이터셋)에도 복사한다.
                  배치 예측 같은 경우엔 False로 호출해 학습 데이터셋을 보존.
    """
    safe_name = Path(filename).name  # 경로 주입 차단
    target = UPLOADS_DIR / safe_name

    written = 0
    chunk = 1024 * 1024
    with target.open('wb') as out:
        while True:
            buf = stream.read(chunk)
            if not buf:
                break
            written += len(buf)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                target.unlink(missing_ok=True)
                raise UploadTooLarge(
                    f'업로드 크기가 제한({MAX_UPLOAD_BYTES // 1024 // 1024}MB)을 초과했습니다.'
                )
            out.write(buf)

    if activate:
        shutil.copyfile(target, DATA_PATH)
    return target, written


def summarize_csv(path: Path) -> DatasetSummary:
    """저장된 CSV의 행/열/결측 요약."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    rows = len(df)

    columns: list[ColumnSummary] = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        columns.append(
            ColumnSummary(
                name=col,
                dtype=str(df[col].dtype),
                missing=miss,
                missing_pct=round(miss / rows * 100, 2) if rows else 0.0,
            )
        )

    return DatasetSummary(
        rows=rows,
        cols=len(df.columns),
        columns=columns,
        has_target=TARGET_COLUMN in df.columns,
    )
