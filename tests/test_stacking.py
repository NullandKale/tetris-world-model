import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from token_world.diagnostics.run_outputs import load_optimizer, optimizer_state
from token_world.models.dynamics import Dynamics, deepen, incoming_actions, masked_loss, patch_pixels, stacked_source
from token_world.models.dynamics import training_mask


def model(layers: int) -> Dynamics:
    return Dynamics(dim=32, layers=layers, heads=2, patch=16, frames=6, colour_dim=8, space_rope=True,
                    action_norm=True, motion=True, motion_first=True, local_attention=True)


def adam(m: Dynamics):
    names = [n for n, _ in m.named_parameters()]
    return torch.optim.AdamW(m.parameters(), lr=1e-3), names


class StackingTests(unittest.TestCase):
    def test_the_whole_model_is_stacked_in_order(self):
        torch.manual_seed(0)
        small, big = model(2), model(6)
        deepen(big, small.state_dict(), stack=3)
        source = small.state_dict()
        for name, value in big.state_dict().items():
            self.assertTrue(torch.equal(value, source[stacked_source(name, 2)]), name)
        self.assertEqual(stacked_source("blocks.5.spatial.local_bias", 2), "blocks.1.spatial.local_bias")
        self.assertEqual(stacked_source("colour_bias", 2), "colour_bias")
        with self.assertRaises(ValueError):
            deepen(model(5), source, stack=3)                                  # 2 blocks stack to 6, not 5

    def test_each_copy_takes_its_source_blocks_optimizer_state(self):
        torch.manual_seed(0)
        small = model(2)
        optimizer, names = adam(small)
        x = torch.randint(0, 20, (1, 6, 256, 256)).to(torch.uint8)
        mask = training_mask(1, 6, 256, "cpu", torch.Generator().manual_seed(0))
        masked_loss(small, small(x, incoming_actions(torch.zeros(1, 6, dtype=torch.long)), mask),
                    patch_pixels(x, 16), mask)[0].backward()
        optimizer.step()
        saved = optimizer_state(optimizer, names)
        big = model(6)
        deepen(big, small.state_dict(), stack=3)
        big_optimizer, big_names = adam(big)
        load_optimizer(big_optimizer, saved, big_names, source_of=lambda n: stacked_source(n, 2))
        small_state = dict(zip(names, (optimizer.state[p] for p in small.parameters())))
        for name, p in zip(big_names, big.parameters()):
            source = small_state[stacked_source(name, 2)]
            if not source:
                continue
            got = big_optimizer.state[p]
            self.assertTrue(torch.equal(got["exp_avg"], source["exp_avg"]), name)
            self.assertTrue(torch.equal(got["exp_avg_sq"], source["exp_avg_sq"]), name)
        copies = [big_optimizer.state[p]["exp_avg"] for n, p in zip(big_names, big.parameters())
                  if n.startswith("blocks.") and n.endswith("spatial.out.weight")]
        copies[0].add_(1.0)
        self.assertFalse(torch.equal(copies[0], copies[2]))                    # copies, not one shared tensor


if __name__ == "__main__":
    unittest.main()
