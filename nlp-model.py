import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import logging
from typing import Optional, List, Tuple
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 5000
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    label_smoothing: float = 0.1

class PositionalEncoding(nn.Module):
    """Inject information about position of tokens in sequence"""
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and restored)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        return self.dropout(x + self.pe[:, :x.size(1)])

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear layers for Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-head attention"""
        batch_size = q.size(0)
        
        # Linear projections and reshape for attention heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        return self.output(attention_output)

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Initialize with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN to input"""
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward network"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process input through self-attention and feed-forward layers"""
        # Self-attention block
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class EnhancedNLPModel(nn.Module):
    """Advanced NLP model with improved architecture and training capabilities"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Output layer
        self.final_layer = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize model parameters with appropriate scaling"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model"""
        # Convert token IDs to embeddings and add positional encoding
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        x = self.positional_encoding(x)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Project to vocabulary size
        return self.final_layer(x)

class EnhancedTrainer:
    """Advanced trainer with mixed precision and gradient accumulation"""
    def __init__(self, model: EnhancedNLPModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        # Mixed precision training
        self.scaler = GradScaler()

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            step = max(1, step)
            arg1 = step ** -0.5
            arg2 = step * (self.config.warmup_steps ** -1.5)
            return min(arg1, arg2)
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader: DataLoader, accumulation_steps: int = 4):
        """Train for one epoch with gradient accumulation and mixed precision"""
        self.model.train()
        total_loss = 0
        
        for idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            # Mixed precision forward pass
            with autocast():
                output = self.model(batch[:, :-1])
                loss = self.criterion(
                    output.view(-1, output.size(-1)),
                    batch[:, 1:].contiguous().view(-1)
                ) / accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            if (idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            if idx % 100 == 0:
                logger.info(f"Batch {idx}, Loss: {loss.item() * accumulation_steps:.4f}")
        
        return total_loss / len(dataloader)

def create_model(vocab_size: int) -> Tuple[EnhancedNLPModel, EnhancedTrainer]:
    """Create model and trainer with default configuration"""
    config = ModelConfig(vocab_size=vocab_size)
    model = EnhancedNLPModel(config)
    trainer = EnhancedTrainer(model, config)
    return model, trainer
