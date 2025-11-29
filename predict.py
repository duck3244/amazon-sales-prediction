"""
학습된 모델을 사용한 예측 스크립트
"""

import torch
import numpy as np
import pandas as pd
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import get_model
from train import SalesDataset
from torch.utils.data import DataLoader


class Predictor:
    """예측 클래스"""
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model (nn.Module): 학습된 모델
            device (str): 디바이스
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, X):
        """
        예측 수행
        
        Args:
            X (np.ndarray): 입력 데이터
        
        Returns:
            np.ndarray: 예측값
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        return predictions
    
    def predict_dataloader(self, dataloader):
        """
        DataLoader를 사용한 배치 예측
        
        Args:
            dataloader: DataLoader 객체
        
        Returns:
            tuple: (predictions, actuals)
        """
        self.model.eval()
        all_predictions = []
        all_actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_actuals.extend(batch_y.numpy().flatten())
        
        return np.array(all_predictions), np.array(all_actuals)


def load_model(model_path, config_path, device='cpu'):
    """
    저장된 모델 로드
    
    Args:
        model_path (str): 모델 체크포인트 경로
        config_path (str): 설정 파일 경로
        device (str): 디바이스
    
    Returns:
        nn.Module: 로드된 모델
    """
    # 설정 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 모델 생성
    model = get_model(
        model_type=config['model_type'],
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout']
    )
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    print(f"Validation Loss: {checkpoint['val_loss']:.6f}")
    
    return model


def evaluate_model(predictions, actuals, save_dir='results'):
    """
    모델 성능 평가
    
    Args:
        predictions (np.ndarray): 예측값
        actuals (np.ndarray): 실제값
        save_dir (str): 결과 저장 디렉토리
    """
    # 디렉토리 생성
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 성능 지표 계산
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    # 결과 출력
    print("\n=== 모델 성능 평가 ===")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # 결과 저장
    results = {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'R2_Score': float(r2)
    }
    
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved: {save_dir / 'evaluation_results.json'}")
    
    # 시각화
    visualize_predictions(predictions, actuals, save_dir)
    
    return results


def visualize_predictions(predictions, actuals, save_dir):
    """
    Visualize prediction results

    Args:
        predictions (np.ndarray): Predicted values
        actuals (np.ndarray): Actual values
        save_dir (Path): Save directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Scatter plot (Actual vs Predicted)
    axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=20)
    axes[0, 0].plot([actuals.min(), actuals.max()],
                    [actuals.min(), actuals.max()],
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Values', fontsize=12)
    axes[0, 0].set_title('Actual vs Predicted', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residual plot
    residuals = actuals - predictions
    axes[0, 1].scatter(predictions, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residual distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Residual Distribution', fontsize=14)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Time series comparison (First 100 samples only)
    sample_size = min(100, len(predictions))
    indices = np.arange(sample_size)
    axes[1, 1].plot(indices, actuals[:sample_size], label='Actual', linewidth=2, marker='o', markersize=4)
    axes[1, 1].plot(indices, predictions[:sample_size], label='Predicted', linewidth=2, marker='x', markersize=4)
    axes[1, 1].set_xlabel('Sample Index', fontsize=12)
    axes[1, 1].set_ylabel('Value', fontsize=12)
    axes[1, 1].set_title('Sample Predictions (First 100)', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'prediction_visualization.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {save_dir / 'prediction_visualization.png'}")
    plt.close()


def save_predictions(predictions, actuals, output_path='predictions.csv'):
    """
    예측 결과를 CSV로 저장

    Args:
        predictions (np.ndarray): 예측값
        actuals (np.ndarray): 실제값
        output_path (str): 출력 경로
    """
    df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Residual': actuals - predictions,
        'Absolute_Error': np.abs(actuals - predictions)
    })

    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved: {output_path}")


def predict_new_data(model, new_data_path, preprocessor_path, device='cpu'):
    """
    새로운 데이터에 대한 예측

    Args:
        model: 학습된 모델
        new_data_path (str): 새 데이터 경로
        preprocessor_path (str): 전처리기 경로
        device (str): 디바이스

    Returns:
        np.ndarray: 예측값
    """
    import pickle

    # 전처리기 로드
    with open(preprocessor_path, 'rb') as f:
        preprocessor_data = pickle.load(f)

    # 새 데이터 로드 및 전처리
    df = pd.read_csv(new_data_path)

    # 전처리 로직 적용 (data_preprocessing.py의 로직과 동일하게)
    # 이 부분은 실제 데이터 구조에 맞게 수정 필요

    # 예측
    predictor = Predictor(model, device)
    predictions = predictor.predict(df.values)

    return predictions


def main(args):
    """메인 실행 함수"""
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 모델 로드
    model = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device
    )

    # Predictor 생성
    predictor = Predictor(model, device)

    # 테스트 데이터 로드
    X_test = np.load(args.test_data)
    y_test = np.load(args.test_labels)

    print(f"\nTest data shape: {X_test.shape}")

    # 예측 수행
    if args.use_dataloader:
        # DataLoader 사용
        test_dataset = SalesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        predictions, actuals = predictor.predict_dataloader(test_loader)
    else:
        # 직접 예측
        predictions = predictor.predict(X_test)
        actuals = y_test

    print(f"Predictions shape: {predictions.shape}")

    # 성능 평가
    results = evaluate_model(predictions, actuals, save_dir=args.save_dir)

    # 예측 결과 저장
    save_predictions(
        predictions,
        actuals,
        output_path=os.path.join(args.save_dir, 'predictions.csv')
    )

    print("\n예측 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Amazon Sales Prediction')

    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default='models/config.json',
                        help='Path to config file')
    parser.add_argument('--test_data', type=str, default='X_test.npy',
                        help='Path to test data')
    parser.add_argument('--test_labels', type=str, default='y_test.npy',
                        help='Path to test labels')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')
    parser.add_argument('--use_dataloader', action='store_true',
                        help='Use DataLoader for prediction')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()

    main(args)