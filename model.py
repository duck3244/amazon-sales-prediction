"""
PyTorch 판매 예측 모델 정의
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SalesPredictor(nn.Module):
    """
    판매량 예측을 위한 신경망 모델
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """
        Args:
            input_dim (int): 입력 특성의 차원
            hidden_dims (list): 은닉층의 차원 리스트
            dropout_rate (float): 드롭아웃 비율
        """
        super(SalesPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 레이어 구성
        layers = []
        
        # 입력층 -> 첫 번째 은닉층
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 은닉층들
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 출력층
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 텐서 [batch_size, input_dim]
        
        Returns:
            torch.Tensor: 예측값 [batch_size, 1]
        """
        return self.network(x)


class ResidualBlock(nn.Module):
    """잔차 연결을 포함한 블록"""
    
    def __init__(self, dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += residual  # 잔차 연결
        out = F.relu(out)
        
        return out


class AdvancedSalesPredictor(nn.Module):
    """
    잔차 연결을 포함한 고급 판매 예측 모델
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_residual_blocks=3, dropout_rate=0.3):
        """
        Args:
            input_dim (int): 입력 특성의 차원
            hidden_dim (int): 은닉층 차원
            num_residual_blocks (int): 잔차 블록의 개수
            dropout_rate (float): 드롭아웃 비율
        """
        super(AdvancedSalesPredictor, self).__init__()
        
        # 입력층
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 잔차 블록들
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate)
            for _ in range(num_residual_blocks)
        ])
        
        # 출력층
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """순전파"""
        x = self.input_layer(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        
        return x


class AttentionSalesPredictor(nn.Module):
    """
    어텐션 메커니즘을 포함한 판매 예측 모델
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, dropout_rate=0.3):
        """
        Args:
            input_dim (int): 입력 특성의 차원
            hidden_dim (int): 은닉층 차원
            num_heads (int): 어텐션 헤드 수
            dropout_rate (float): 드롭아웃 비율
        """
        super(AttentionSalesPredictor, self).__init__()
        
        # 특성 임베딩
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-head Self-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward 네트워크
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 출력층
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """순전파"""
        # 특성 임베딩
        x = self.feature_embedding(x)
        
        # Self-Attention (배치 차원 추가)
        x_unsqueezed = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attn_output, _ = self.multihead_attn(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # 잔차 연결 및 정규화
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        # 출력
        output = self.output_layer(x)
        
        return output


def get_model(model_type='basic', input_dim=10, **kwargs):
    """
    모델 생성 팩토리 함수
    
    Args:
        model_type (str): 모델 타입 ('basic', 'advanced', 'attention')
        input_dim (int): 입력 차원
        **kwargs: 모델별 추가 파라미터
    
    Returns:
        nn.Module: PyTorch 모델
    """
    if model_type == 'basic':
        return SalesPredictor(input_dim, **kwargs)
    elif model_type == 'advanced':
        return AdvancedSalesPredictor(input_dim, **kwargs)
    elif model_type == 'attention':
        return AttentionSalesPredictor(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 모델 테스트
    print("=== 모델 테스트 ===\n")
    
    input_dim = 20
    batch_size = 32
    
    # 더미 입력 데이터
    dummy_input = torch.randn(batch_size, input_dim)
    
    # Basic 모델
    print("1. Basic Model:")
    model_basic = get_model('basic', input_dim=input_dim)
    output = model_basic(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_basic.parameters()):,}\n")
    
    # Advanced 모델
    print("2. Advanced Model (with Residual Blocks):")
    model_advanced = get_model('advanced', input_dim=input_dim)
    output = model_advanced(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_advanced.parameters()):,}\n")
    
    # Attention 모델
    print("3. Attention Model:")
    model_attention = get_model('attention', input_dim=input_dim)
    output = model_attention(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_attention.parameters()):,}")
