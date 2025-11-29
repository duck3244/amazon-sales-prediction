"""
데이터 전처리 파이프라인
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = None
        
    def load_data(self, file_path):
        """데이터 로드"""
        print(f"데이터 로딩: {file_path}")
        df = pd.read_csv(file_path)
        print(f"원본 데이터 shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """결측치 처리"""
        print("\n결측치 처리 중...")
        
        # 수치형 변수: 중앙값으로 채우기
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"  {col}: 중앙값({median_value})으로 대체")
        
        # 범주형 변수: 최빈값으로 채우기
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                print(f"  {col}: 최빈값({mode_value})으로 대체")
        
        return df
    
    def extract_date_features(self, df, date_columns):
        """날짜 특성 추출"""
        print("\n날짜 특성 추출 중...")
        
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    
                    print(f"  {col}에서 5개 특성 생성")
                    
                    # 원본 날짜 컬럼 제거
                    df.drop(col, axis=1, inplace=True)
                except Exception as e:
                    print(f"  {col} 처리 실패: {e}")
        
        return df
    
    def encode_categorical_features(self, df, categorical_columns=None):
        """범주형 변수 인코딩"""
        print("\n범주형 변수 인코딩 중...")
        
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col in df.columns:
                # 카테고리 수가 너무 많으면 상위 N개만 유지
                if df[col].nunique() > 50:
                    top_categories = df[col].value_counts().head(50).index
                    df[col] = df[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                    print(f"  {col}: 상위 50개 카테고리만 유지")
                
                # Label Encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: Label Encoding 완료 ({len(le.classes_)}개 클래스)")
        
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """이상치 제거"""
        print("\n이상치 제거 중...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        original_len = len(df)
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                before_len = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = before_len - len(df)
                
                if removed > 0:
                    print(f"  {col}: {removed}개 행 제거")
        
        print(f"전체 제거된 행: {original_len - len(df)}")
        return df
    
    def create_features(self, df):
        """추가 특성 생성"""
        print("\n추가 특성 생성 중...")
        
        # 예시: Amount와 Quantity가 있다면 평균 가격 계산
        if 'Amount' in df.columns and 'Qty' in df.columns:
            df['avg_price'] = df['Amount'] / (df['Qty'] + 1)  # 0으로 나누기 방지
            print("  avg_price 특성 생성")
        
        # 추가 특성은 데이터에 따라 커스터마이징
        
        return df
    
    def normalize_features(self, df, exclude_columns=None):
        """특성 정규화"""
        print("\n특성 정규화 중...")
        
        if exclude_columns is None:
            exclude_columns = []
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_normalize = [col for col in numerical_cols if col not in exclude_columns]
        
        if len(cols_to_normalize) > 0:
            df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
            print(f"  {len(cols_to_normalize)}개 컬럼 정규화 완료")
        
        return df
    
    def prepare_train_test_split(self, df, target_column, test_size=0.2, val_size=0.1):
        """학습/검증/테스트 데이터 분할"""
        print("\n데이터 분할 중...")
        
        self.target_column = target_column
        
        # 타겟 분리
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        
        # Train/Temp 분할
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=42
        )
        
        # Validation/Test 분할
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=42
        )
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, file_path='preprocessor.pkl'):
        """전처리 객체 저장"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'target_column': self.target_column
            }, f)
        print(f"\n전처리 객체 저장: {file_path}")
    
    def load_preprocessor(self, file_path='preprocessor.pkl'):
        """전처리 객체 로드"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoders = data['label_encoders']
            self.feature_names = data['feature_names']
            self.target_column = data['target_column']
        print(f"전처리 객체 로드: {file_path}")

def main():
    """메인 실행 함수"""
    # 데이터 파일 경로
    data_path = "Amazon Sale Report.csv"

    # 전처리 파이프라인 실행
    preprocessor = DataPreprocessor()

    try:
        # 1. 데이터 로드
        df = preprocessor.load_data(data_path)

        # 2. 결측치 처리
        df = preprocessor.handle_missing_values(df)

        # 3. 날짜 특성 추출 (데이터에 날짜 컬럼이 있다면)
        date_columns = ['Date', 'ship-date', 'Order Date']  # 실제 컬럼명으로 수정
        df = preprocessor.extract_date_features(df, date_columns)

        # 4. 범주형 변수 인코딩
        df = preprocessor.encode_categorical_features(df)

        # 5. 추가 특성 생성
        df = preprocessor.create_features(df)

        # 6. 타겟 변수 설정 (예: Amount를 예측)
        # 실제 데이터에 맞게 수정 필요
        target_column = 'Amount'  # 또는 'Qty', 'Price' 등

        if target_column not in df.columns:
            print(f"\n경고: '{target_column}' 컬럼을 찾을 수 없습니다.")
            print("사용 가능한 컬럼:", df.columns.tolist())
            return

        # 7. 이상치 제거 (타겟 제외)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(target_column)
        df = preprocessor.remove_outliers(df, columns=numeric_cols)

        # 8. 특성 정규화 (타겟 제외)
        df = preprocessor.normalize_features(df, exclude_columns=[target_column])

        # 9. 학습/검증/테스트 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = \
            preprocessor.prepare_train_test_split(df, target_column)

        # 10. 전처리된 데이터 저장
        # 수치형으로 명시적 변환
        np.save('X_train.npy', X_train.values.astype(np.float32))
        np.save('X_val.npy', X_val.values.astype(np.float32))
        np.save('X_test.npy', X_test.values.astype(np.float32))
        np.save('y_train.npy', y_train.values.astype(np.float32))
        np.save('y_val.npy', y_val.values.astype(np.float32))
        np.save('y_test.npy', y_test.values.astype(np.float32))
        
        print("\n전처리된 데이터 저장 완료:")
        print("  - X_train.npy, y_train.npy")
        print("  - X_val.npy, y_val.npy")
        print("  - X_test.npy, y_test.npy")
        
        # 11. 전처리 객체 저장
        preprocessor.save_preprocessor()
        
        print("\n데이터 전처리 완료!")
        print(f"특성 개수: {len(preprocessor.feature_names)}")
        print(f"타겟 변수: {target_column}")
        
    except FileNotFoundError:
        print(f"오류: '{data_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
