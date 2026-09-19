import unittest

import torch

from config import ModelConfig
from model import MiniLLM


class MiniLLMTestCase(unittest.TestCase):
    def test_model_forward_shape(self):
        cfg = ModelConfig(n_layers=2, d_model=64, n_heads=4, kv_heads=4, max_seq_len=128, vocab_size=200)
        model = MiniLLM(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (2, 32, cfg.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_masked_loss_accepts_minus_one(self):
        cfg = ModelConfig(n_layers=2, d_model=64, n_heads=4, kv_heads=4, max_seq_len=128, vocab_size=200)
        model = MiniLLM(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = x.clone()
        y[:, -2:] = -1
        logits, loss = model(x, y)
        self.assertTrue(torch.isfinite(loss))

    def test_causal_mask_is_effective(self):
        cfg = ModelConfig(n_layers=2, d_model=64, n_heads=4, kv_heads=4, max_seq_len=128, vocab_size=200)
        model = MiniLLM(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _ = model(x)
        self.assertEqual(logits.shape, (1, 1, cfg.vocab_size))


if __name__ == "__main__":
    unittest.main()
