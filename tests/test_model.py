import torch
from NLPModel import create_model, ModelConfig

def test_model_initialization():
    vocab_size = 1000
    model, trainer = create_model(vocab_size)
    assert model is not None
    assert trainer is not None

def test_forward_pass():
    vocab_size = 1000
    model, _ = create_model(vocab_size)
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    assert output.shape == (batch_size, seq_len, vocab_size)
