import unittest
import torch
from torch.nn import functional as F
from distance_alpha import DistanceAlpha


class MemoryTests(unittest.TestCase):
    def test_causality_and_gradients(self):
        torch.manual_seed(7)
        m = DistanceAlpha(vocab_size=32, block_size=8, dim=16, heads=2,
                          layers=1, clusters=4, slots=4).eval()
        x = torch.randint(32, (2, 8))
        altered = x.clone(); altered[:, 4:] = (altered[:, 4:] + 1) % 32
        logits, aux, effective = m(x)
        other, _, _ = m(altered)
        torch.testing.assert_close(logits[:, :4], other[:, :4])
        self.assertTrue(1 <= effective.item() <= 16.001)
        (logits.square().mean() + .001*aux).backward()
        for p in m.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertIs(m.output.weight, m.token.weight)

    def test_language_model_initial_loss_is_not_exploded(self):
        torch.manual_seed(11)
        m = DistanceAlpha(vocab_size=64, block_size=8, dim=16, heads=2,
                          layers=1, clusters=4, slots=4)
        x = torch.randint(64, (1, 8))
        logits, _, _ = m(x)
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), x[:, 1:].flatten())
        self.assertLess(loss.item(), 8.)


if __name__ == '__main__': unittest.main()
