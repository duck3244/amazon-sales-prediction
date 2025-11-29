"""
PyTorch 모델 학습 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from model import get_model


class SalesDataset(Dataset):
    """판매 데이터 Dataset 클래스"""
    
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): 특성 데이터
            y (np.ndarray): 타겟 데이터
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early Stopping 클래스"""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience (int): 개선이 없을 때 기다리는 에포크 수
            min_delta (float): 개선으로 간주할 최소 변화량
            mode (str): 'min' (손실) 또는 'max' (정확도)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class Trainer:
    """모델 학습 클래스"""
    
    def __init__(self, model, device, save_dir='models'):
        """
        Args:
            model (nn.Module): PyTorch 모델
            device (str): 디바이스 ('cuda' 또는 'cpu')
            save_dir (str): 모델 저장 디렉토리
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader, criterion, optimizer):
        """한 에포크 학습"""
        self.model.train()
        epoch_loss = 0
        
        with tqdm(dataloader, desc='Training') as pbar:
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """검증"""
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
    
    def fit(self, train_loader, val_loader, criterion, optimizer, 
            scheduler=None, epochs=100, early_stopping=None):
        """
        모델 학습
        
        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            criterion: 손실 함수
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러
            epochs (int): 에포크 수
            early_stopping: Early Stopping 객체
        """
        print("\n=== 학습 시작 ===")
        
        for epoch in range(epochs):
            # 학습
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            
            # 검증
            val_loss = self.validate(val_loader, criterion)
            
            # 손실 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 스케줄러 업데이트
            if scheduler is not None:
                scheduler.step(val_loss)
            
            # 로그 출력
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            
            # 최고 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(f"  ✓ Best model saved (Val Loss: {val_loss:.6f})")
            
            # Early Stopping 체크
            if early_stopping is not None:
                if early_stopping(val_loss):
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print("\n=== 학습 완료 ===")
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename):
        """체크포인트 로드"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filename}")
        return checkpoint
    
    def plot_losses(self, save_path='training_history.png'):
        """Save loss graph"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training History', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved: {save_path}")
        plt.close()


def load_data(data_dir='.'):
    """전처리된 데이터 로드"""
    print("데이터 로딩 중...")

    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main(args):
    """메인 실행 함수"""
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 재현성을 위한 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 데이터 로드
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Dataset 및 DataLoader 생성
    train_dataset = SalesDataset(X_train, y_train)
    val_dataset = SalesDataset(X_val, y_val)
    test_dataset = SalesDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 모델 생성
    input_dim = X_train.shape[1]
    model = get_model(
        model_type=args.model_type,
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout
    )

    print(f"\nModel: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')

    # Trainer 생성 및 학습
    trainer = Trainer(model, device, save_dir=args.save_dir)
    trainer.fit(
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        early_stopping=early_stopping
    )

    # 학습 히스토리 저장
    trainer.plot_losses(os.path.join(args.save_dir, 'training_history.png'))

    # 학습 설정 저장
    config = {
        'model_type': args.model_type,
        'input_dim': input_dim,
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'best_val_loss': trainer.best_val_loss,
    }

    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\nConfig saved: {args.save_dir}/config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Amazon Sales Prediction Training')

    # 모델 파라미터
    parser.add_argument('--model_type', type=str, default='basic',
                        choices=['basic', 'advanced', 'attention'],
                        help='Model type')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    # 기타
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Model save directory')

    args = parser.parse_args()

    main(args)