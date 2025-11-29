"""
유틸리티 함수 모음
"""

import torch
import numpy as np
import pandas as pd
import random
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed=42):
    """
    재현성을 위한 시드 설정
    
    Args:
        seed (int): 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def count_parameters(model):
    """
    모델의 파라미터 개수 계산
    
    Args:
        model (nn.Module): PyTorch 모델
    
    Returns:
        dict: 학습 가능/불가능 파라미터 개수
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    
    return {
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total': total
    }


def print_model_summary(model, input_shape=None):
    """
    모델 요약 정보 출력
    
    Args:
        model (nn.Module): PyTorch 모델
        input_shape (tuple): 입력 shape (배치 제외)
    """
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    params = count_parameters(model)
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    print(f"Total parameters: {params['total']:,}")
    print("=" * 70)
    
    if input_shape is not None:
        dummy_input = torch.randn(1, *input_shape)
        try:
            output = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print("=" * 70)
        except Exception as e:
            print(f"Could not perform forward pass: {e}")


def save_dict_to_json(data_dict, file_path):
    """
    딕셔너리를 JSON 파일로 저장
    
    Args:
        data_dict (dict): 저장할 딕셔너리
        file_path (str): 저장 경로
    """
    with open(file_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"Dictionary saved to {file_path}")


def load_dict_from_json(file_path):
    """
    JSON 파일에서 딕셔너리 로드
    
    Args:
        file_path (str): JSON 파일 경로
    
    Returns:
        dict: 로드된 딕셔너리
    """
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    print(f"Dictionary loaded from {file_path}")
    return data_dict


def create_directories(dirs):
    """
    디렉토리 생성
    
    Args:
        dirs (list): 생성할 디렉토리 리스트
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {dir_path}")


def plot_feature_importance(feature_names, importance_values, top_n=20, save_path=None):
    """
    Visualize feature importance

    Args:
        feature_names (list): List of feature names
        importance_values (np.ndarray): Importance values
        top_n (int): Number of top features to display
        save_path (str): Save path
    """
    # Sort feature importance
    indices = np.argsort(importance_values)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importance_values[indices]

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_values, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {save_path}")

    plt.show()
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    학습 곡선 시각화

    Args:
        train_losses (list): 학습 손실 리스트
        val_losses (list): 검증 손실 리스트
        save_path (str): 저장 경로
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")

    plt.show()
    plt.close()


def analyze_predictions(y_true, y_pred, save_dir=None):
    """
    Comprehensive prediction analysis

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        save_dir (str): Save directory
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Calculate performance metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Calculate residuals
    residuals = y_true - y_pred

    # Statistics
    stats = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Mean Residual': np.mean(residuals),
        'Std Residual': np.std(residuals),
        'Min Residual': np.min(residuals),
        'Max Residual': np.max(residuals)
    }

    # Print
    print("\n=== Prediction Analysis ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title('True vs Predicted Values')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residual plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residual distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, 'prediction_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nAnalysis plot saved: {save_path}")

    plt.show()
    plt.close()

    return stats


def compute_class_weights(y, num_classes=None):
    """
    클래스 불균형 처리를 위한 가중치 계산

    Args:
        y (np.ndarray): 타겟 배열
        num_classes (int): 클래스 개수

    Returns:
        torch.Tensor: 클래스 가중치
    """
    from sklearn.utils.class_weight import compute_class_weight

    if num_classes is None:
        num_classes = len(np.unique(y))

    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=y
    )

    return torch.FloatTensor(class_weights)


def normalize_data(data, method='standardize'):
    """
    데이터 정규화

    Args:
        data (np.ndarray): 입력 데이터
        method (str): 정규화 방법 ('standardize' 또는 'minmax')

    Returns:
        tuple: (정규화된 데이터, 파라미터)
    """
    if method == 'standardize':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def denormalize_data(data, params, method='standardize'):
    """
    정규화 역변환

    Args:
        data (np.ndarray): 정규화된 데이터
        params (dict): 정규화 파라미터
        method (str): 정규화 방법

    Returns:
        np.ndarray: 원본 스케일 데이터
    """
    if method == 'standardize':
        denormalized = data * params['std'] + params['mean']
    elif method == 'minmax':
        denormalized = data * (params['max'] - params['min']) + params['min']
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return denormalized


def get_device(gpu_id=0):
    """
    사용 가능한 디바이스 반환

    Args:
        gpu_id (int): GPU ID

    Returns:
        torch.device: 디바이스
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def print_system_info():
    """시스템 정보 출력"""
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print("=" * 50)


if __name__ == "__main__":
    # 유틸리티 함수 테스트
    print_system_info()

    # 시드 설정 테스트
    set_seed(42)

    # 디바이스 테스트
    device = get_device()

    print("\nUtility functions are ready to use!")