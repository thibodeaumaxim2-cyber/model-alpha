import unittest
import torch
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


if __name__ == '__main__': unittest.main()
