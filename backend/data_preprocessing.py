"""데이터 전처리 파이프라인

구조 원칙:
  - target leakage 방지: 타겟(`Amount`)을 사용한 파생 변수를 만들지 않는다.
  - 데이터 누수 방지: split 이후에 train으로만 fit하고, val/test는 transform만 수행한다.
  - 타겟 결측 행은 drop한다(중앙값으로 채워서 학습에 사용 금지).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# 사전적으로 제거할 컬럼 (id/노이즈/거의 단일값)
DROP_COLUMNS_DEFAULT = [
    'index',          # 행 번호 (인공)
    'Order ID',       # 거의 unique
    'SKU',            # 거의 unique
    'ASIN',           # 거의 unique
    'Unnamed: 22',    # 빈 컬럼
    'currency',       # 거의 단일값(INR)
]


class DataPreprocessor:
    """데이터 전처리 클래스

    fit/transform 패턴을 따르며, 인코더/스케일러는 train 데이터로만 fit한다.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}        # col -> LabelEncoder
        self.category_whitelist = {}    # col -> set(상위 카테고리). 그 외는 'Other'
        self.numeric_medians = {}       # 수치형 결측치 채울 train 중앙값
        self.categorical_modes = {}     # 범주형 결측치 채울 train 최빈값
        self.feature_names = []
        self.target_column = None
        self._normalize_cols = []

    # ---------- I/O ----------
    def load_data(self, file_path):
        print(f"데이터 로딩: {file_path}")
        df = pd.read_csv(file_path)
        # 트레일링 공백이 있는 컬럼명 정리 (예: 'Sales Channel ')
        df.columns = [c.strip() for c in df.columns]
        print(f"원본 데이터 shape: {df.shape}")
        return df

    # ---------- 컬럼 정리 ----------
    @staticmethod
    def drop_unwanted_columns(df, columns=None):
        if columns is None:
            columns = DROP_COLUMNS_DEFAULT
        existing = [c for c in columns if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
            print(f"제거 컬럼: {existing}")
        return df

    @staticmethod
    def drop_target_missing(df, target_column):
        before = len(df)
        df = df.dropna(subset=[target_column]).reset_index(drop=True)
        print(f"타겟 결측 행 제거: {before - len(df)}개")
        return df

    @staticmethod
    def filter_status(df, status_column='Status', drop_values=('Cancelled',)):
        """취소 등 비정상 주문 제거 (Amount=0이 다수)."""
        if status_column not in df.columns:
            return df
        mask = ~df[status_column].isin(drop_values)
        before = len(df)
        df = df[mask].reset_index(drop=True)
        print(f"Status 필터링({drop_values}): {before - len(df)}개 제거")
        return df

    # ---------- 결측치 ----------
    def fit_missing(self, df_train):
        for col in df_train.select_dtypes(include=[np.number]).columns:
            self.numeric_medians[col] = df_train[col].median()
        for col in df_train.select_dtypes(include=['object']).columns:
            mode = df_train[col].mode()
            self.categorical_modes[col] = mode[0] if len(mode) > 0 else 'Unknown'

    def transform_missing(self, df):
        for col, val in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col, val in self.categorical_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    # ---------- 날짜 ----------
    @staticmethod
    def extract_date_features(df, date_columns):
        for col in date_columns:
            if col not in df.columns:
                continue
            try:
                dt = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_year'] = dt.dt.year
                df[f'{col}_month'] = dt.dt.month
                df[f'{col}_day'] = dt.dt.day
                df[f'{col}_dayofweek'] = dt.dt.dayofweek
                df[f'{col}_quarter'] = dt.dt.quarter
                df = df.drop(columns=[col])
            except Exception as e:
                print(f"  {col} 처리 실패: {e}")
        return df

    # ---------- 특수 컬럼: promotion-ids → boolean ----------
    @staticmethod
    def transform_promotion_ids(df, col='promotion-ids'):
        if col in df.columns:
            df[f'{col}_present'] = df[col].notna().astype(int)
            df = df.drop(columns=[col])
        return df

    # ---------- 범주형 ----------
    def fit_categorical(self, df_train, columns=None, top_n=50):
        if columns is None:
            columns = df_train.select_dtypes(include=['object']).columns.tolist()
        for col in columns:
            if col not in df_train.columns:
                continue
            series = df_train[col].astype(str)
            if series.nunique() > top_n:
                whitelist = set(series.value_counts().head(top_n).index)
                self.category_whitelist[col] = whitelist
                series = series.where(series.isin(whitelist), 'Other')
            le = LabelEncoder()
            # 'Unknown'/'Other'를 미리 포함시켜 transform 시 미지 카테고리 대응
            classes = list(set(series.tolist()) | {'Unknown', 'Other'})
            le.fit(classes)
            self.label_encoders[col] = le

    def transform_categorical(self, df):
        for col, le in self.label_encoders.items():
            if col not in df.columns:
                continue
            series = df[col].astype(str)
            if col in self.category_whitelist:
                series = series.where(series.isin(self.category_whitelist[col]), 'Other')
            # train에서 본 적 없는 라벨은 'Unknown'으로 매핑
            known = set(le.classes_)
            series = series.where(series.isin(known), 'Unknown')
            df[col] = le.transform(series)
        return df

    # ---------- 정규화 ----------
    def fit_scaler(self, df_train, exclude_columns=None):
        exclude = set(exclude_columns or [])
        self._normalize_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns
                                if c not in exclude]
        if self._normalize_cols:
            self.scaler.fit(df_train[self._normalize_cols])

    def transform_scaler(self, df):
        if self._normalize_cols:
            cols = [c for c in self._normalize_cols if c in df.columns]
            df[cols] = self.scaler.transform(df[cols])
        return df

    # ---------- 통합 transform (예측 경로) ----------
    def transform(self, df, date_columns=None, status_filter=True):
        """학습 시 fit한 통계로 새로운 데이터를 변환한다."""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        df = self.drop_unwanted_columns(df)
        if status_filter:
            df = self.filter_status(df)
        df = self.transform_promotion_ids(df)
        if date_columns:
            df = self.extract_date_features(df, date_columns)
        df = self.transform_missing(df)
        df = self.transform_categorical(df)
        # feature 정렬
        df = df[[c for c in self.feature_names if c in df.columns]]
        df = self.transform_scaler(df)
        return df

    # ---------- save/load ----------
    def save_preprocessor(self, file_path='preprocessor.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"전처리 객체 저장: {file_path}")

    def load_preprocessor(self, file_path='preprocessor.pkl'):
        with open(file_path, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        print(f"전처리 객체 로드: {file_path}")


def main():
    data_path = "Amazon Sale Report.csv"
    target_column = 'Amount'
    date_columns = ['Date']

    pp = DataPreprocessor()

    try:
        # 1) 로드
        df = pp.load_data(data_path)

        # 2) 컬럼 정리 (id/노이즈/단일값)
        df = pp.drop_unwanted_columns(df)

        # 3) 타겟 결측 행 drop (imputation 금지)
        df = pp.drop_target_missing(df, target_column)

        # 4) 비정상 주문 제거 (취소된 주문은 Amount=0이 다수 — 분포 왜곡 유발)
        df = pp.filter_status(df)

        # 5) promotion-ids → 존재 여부 boolean
        df = pp.transform_promotion_ids(df)

        # 6) 날짜 특성 추출
        df = pp.extract_date_features(df, date_columns)

        # ⚠ 주의: avg_price = Amount/Qty 같은 target leakage 파생은 만들지 않는다.

        # 7) 학습/검증/테스트 분할 — fit은 split **이후** train으로만
        X = df.drop(columns=[target_column])
        y = df[target_column].astype(np.float32)
        pp.feature_names = X.columns.tolist()
        pp.target_column = target_column

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.125, random_state=42)  # 0.125 * 0.8 = 0.1

        print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # 8) 결측치/범주형/스케일러 fit (train)
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

        # 9) 저장
        np.save('X_train.npy', X_train.values.astype(np.float32))
        np.save('X_val.npy', X_val.values.astype(np.float32))
        np.save('X_test.npy', X_test.values.astype(np.float32))
        np.save('y_train.npy', y_train.values.astype(np.float32))
        np.save('y_val.npy', y_val.values.astype(np.float32))
        np.save('y_test.npy', y_test.values.astype(np.float32))

        pp.save_preprocessor()

        print("\n전처리 완료")
        print(f"특성 수: {len(pp.feature_names)}, 타겟: {target_column}")

    except FileNotFoundError:
        print(f"오류: '{data_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        import traceback
        print(f"오류 발생: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
