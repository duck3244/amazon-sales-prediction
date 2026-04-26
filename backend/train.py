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
from utils import set_seed


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
    
    def __init__(self, model, device, save_dir='models', quiet=False):
        """
        Args:
            quiet (bool): True면 tqdm 진행률 막대를 비활성화. API 컨텍스트에서
                stdout이 캡처되는 경우 \r 갱신이 노이즈가 되므로 권장.
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.quiet = quiet

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, dataloader, criterion, optimizer):
        """한 에포크 학습"""
        self.model.train()
        epoch_loss = 0

        with tqdm(dataloader, desc='Training', disable=self.quiet) as pbar:
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
            scheduler=None, epochs=100, early_stopping=None, on_epoch_end=None):
        """모델 학습.

        Args:
            on_epoch_end: 매 epoch 종료 시 호출되는 콜백.
                시그니처 ``(epoch:int, total:int, train_loss:float, val_loss:float,
                best_val_loss:float, stopped:bool) -> None``.
                API에서 진행률/loss 곡선 업데이트에 사용.
        """
        print("\n=== 학습 시작 ===")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if scheduler is not None:
                scheduler.step(val_loss)

            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', epoch, val_loss)
                print(f"  ✓ Best model saved (Val Loss: {val_loss:.6f})")

            stopped = False
            if early_stopping is not None and early_stopping(val_loss):
                stopped = True

            if on_epoch_end is not None:
                try:
                    on_epoch_end(epoch + 1, epochs, train_loss, val_loss,
                                 self.best_val_loss, stopped)
                except Exception as cb_err:  # 콜백 오류가 학습을 중단시키지 않도록
                    print(f"  [warn] on_epoch_end 콜백 오류 무시: {cb_err}")

            if stopped:
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
        checkpoint = torch.load(self.save_dir / filename,
                                map_location=self.device, weights_only=False)
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

    # 재현성을 위한 시드 설정 (CUDA, cudnn, hash까지 포함)
    set_seed(args.seed)

    # 데이터 로드
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Dataset 및 DataLoader 생성
    train_dataset = SalesDataset(X_train, y_train)
    val_dataset = SalesDataset(X_val, y_val)
    test_dataset = SalesDataset(X_test, y_test)

    use_cuda = device.type == 'cuda'
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, **loader_kwargs)

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

    # 학습률 스케줄러 (PyTorch 2.4+: verbose 인자는 deprecated)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

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

    # best 가중치 복원 후 test 평가 (학습 마지막 epoch 가중치가 아닌 best로 평가)
    trainer.load_checkpoint('best_model.pth')
    test_loss = trainer.validate(test_loader, criterion)
    print(f"\nTest Loss (best ckpt): {test_loss:.6f}")

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
        'test_loss': test_loss,
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
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader worker processes')

    args = parser.parse_args()

    main(args)