import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
@dataclass
class ModelConfig:
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
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    pad_token_id: int = 0

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.d_k).transpose(1, 2), qkv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)

# --- Feed Forward Network ---
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Transformer Block (Pre-Norm) ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# --- Main NLP Model ---
class NLPModel(nn.Module):  # Renamed from EnhancedNLPModel
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.pos_enc = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config.d_model, config.num_heads, config.d_ff, config.dropout) for _ in range(config.num_layers)]
        )
        self.final_layer = nn.Linear(config.d_model, config.vocab_size)
        self.final_layer.weight = self.embedding.weight  # Weight tying
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.final_layer(x)

# --- Enhanced Trainer ---
class EnhancedTrainer:
    def __init__(self, model: NLPModel, config: ModelConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing, ignore_index=config.pad_token_id
        )
        self.scaler = GradScaler()

    def _create_scheduler(self):
        def lr_lambda(step):
            step = max(1, step)
            return min(step ** -0.5, step * self.config.warmup_steps ** -1.5)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader: DataLoader, accumulation_steps: int = 4):
        self.model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            src, tgt = batch[:, :-1], batch[:, 1:]
            src, tgt = src.to(self.device), tgt.to(self.device)
            mask = torch.tril(torch.ones(src.size(1), src.size(1))).bool().to(src.device)

            with autocast():
                logits = self.model(src, mask)
                loss = self.criterion(logits.view(-1, self.config.vocab_size), tgt.view(-1)) / accumulation_steps

            self.scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            if i % 100 == 0:
                logger.info(f"Batch {i}, Loss: {loss.item() * accumulation_steps:.4f}")
        return total_loss / len(dataloader)

# --- Helper Function ---
def create_model(vocab_size: int) -> tuple[NLPModel, EnhancedTrainer]:
    config = ModelConfig(vocab_size=vocab_size)
    model = NLPModel(config)
    trainer = EnhancedTrainer(model, config)
    return model, trainer
