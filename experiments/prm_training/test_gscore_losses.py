"""CPU checks for CE compatibility, normalized MSE and null-label masking."""
import unittest
import torch
from train_wholecot_prm_gscore import loss_for


class LossTests(unittest.TestCase):
    def test_mask_and_gradient(self):
        for objective in ("ce", "mse"):
            g = torch.zeros((1, 3, 5) if objective == "ce" else (1, 3), requires_grad=True)
            f = torch.zeros((1, 3), requires_grad=True)
            batch = {"labels": {"G": torch.tensor([[0., 4., 2.]]), "F": torch.tensor([[0., 1., 0.]])},
                     "masks": {"G": torch.tensor([[1., 1., 0.]]), "F": torch.tensor([[1., 1., 0.]])}}
            loss, parts = loss_for({"G": g, "F": f}, batch, {"G": 2., "F": 2.}, 30., objective)
            self.assertTrue(torch.isfinite(loss))
            if objective == "mse":
                self.assertAlmostEqual(parts["G"].item(), .25)
            else:
                self.assertAlmostEqual(parts["G"].item(), torch.log(torch.tensor(5.)).item())
            loss.backward()
            self.assertEqual(g.grad[0, 2].abs().sum().item(), 0.)
            self.assertEqual(f.grad[0, 2].item(), 0.)
            if objective == "mse":
                self.assertGreater(g.grad[0, 0].item(), 0.)
                self.assertLess(g.grad[0, 1].item(), 0.)


if __name__ == "__main__":
    unittest.main()
