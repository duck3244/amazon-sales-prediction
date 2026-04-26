"""
Amazon 판매 데이터 탐색 및 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(file_path):
    """데이터 로드"""
    print(f"데이터 로딩 중: {file_path}")
    df = pd.read_csv(file_path)
    print(f"데이터 shape: {df.shape}")
    return df

def explore_data(df):
    """기본 데이터 탐색"""
    print("\n=== 데이터 기본 정보 ===")
    print(df.info())
    
    print("\n=== 첫 5행 ===")
    print(df.head())
    
    print("\n=== 기술 통계 ===")
    print(df.describe())
    
    print("\n=== 결측치 확인 ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        '결측치 개수': missing,
        '비율(%)': missing_pct
    })
    print(missing_df[missing_df['결측치 개수'] > 0])
    
    return df

def analyze_numerical_features(df):
    """Analyze numerical features"""
    print("\n=== Numerical Features Analysis ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) > 0:
        fig, axes = plt.subplots(
            nrows=(len(numerical_cols) + 2) // 3,
            ncols=3,
            figsize=(15, 5 * ((len(numerical_cols) + 2) // 3))
        )
        axes = axes.flatten() if len(numerical_cols) > 1 else [axes]

        for idx, col in enumerate(numerical_cols):
            df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'{col} Distribution')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

        # Remove empty subplots
        for idx in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig('numerical_features_distribution.png', dpi=300, bbox_inches='tight')
        print("Numerical features distribution saved: numerical_features_distribution.png")
        plt.close()

def analyze_categorical_features(df, top_n=10):
    """Analyze categorical features"""
    print("\n=== Categorical Features Analysis ===")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols[:5]:  # Analyze top 5 only
        print(f"\n{col} - Unique values: {df[col].nunique()}")
        print(df[col].value_counts().head(top_n))

        if df[col].nunique() <= 20:
            plt.figure(figsize=(12, 6))
            df[col].value_counts().head(top_n).plot(kind='bar')
            plt.title(f'{col} Distribution (Top {top_n})')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{col}_distribution.png', dpi=300, bbox_inches='tight')
            print(f"{col} distribution saved: {col}_distribution.png")
            plt.close()

def analyze_correlations(df):
    """Correlation analysis"""
    print("\n=== Correlation Analysis ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1
        )
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved: correlation_heatmap.png")
        plt.close()

        print("\nTop Correlations:")
        # Remove diagonal and display top correlations
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1]
        print(corr_pairs.abs().sort_values(ascending=False).head(10))

def detect_outliers(df):
    """Outlier detection"""
    print("\n=== Outlier Detection (IQR Method) ===")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_summary = {}

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100

        outlier_summary[col] = {
            'Outlier_Count': outlier_count,
            'Percentage(%)': outlier_pct,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        }

    outlier_df = pd.DataFrame(outlier_summary).T
    print(outlier_df)

def save_data_summary(df, output_path='data_summary.txt'):
    """데이터 요약 정보 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== Amazon 판매 데이터 요약 ===\n\n")
        f.write(f"전체 행 수: {len(df)}\n")
        f.write(f"전체 열 수: {len(df.columns)}\n\n")

        f.write("=== 컬럼 목록 ===\n")
        for col in df.columns:
            f.write(f"- {col} ({df[col].dtype})\n")

        f.write("\n=== 결측치 정보 ===\n")
        missing = df.isnull().sum()
        for col, count in missing[missing > 0].items():
            f.write(f"- {col}: {count} ({count/len(df)*100:.2f}%)\n")

    print(f"\n데이터 요약 저장: {output_path}")

def main():
    """메인 실행 함수"""
    # 데이터 파일 경로 (실제 경로로 수정 필요)
    data_path = "Amazon Sale Report.csv"

    try:
        # 데이터 로드
        df = load_data(data_path)

        # 데이터 탐색
        df = explore_data(df)

        # 수치형 변수 분석
        analyze_numerical_features(df)

        # 범주형 변수 분석
        analyze_categorical_features(df)

        # 상관관계 분석
        analyze_correlations(df)

        # 이상치 탐지
        detect_outliers(df)

        # 데이터 요약 저장
        save_data_summary(df)

        print("\n데이터 분석 완료!")

    except FileNotFoundError:
        print(f"오류: '{data_path}' 파일을 찾을 수 없습니다.")
        print("Kaggle에서 데이터를 다운로드하고 현재 디렉토리에 저장하세요.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()