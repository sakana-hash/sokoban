"""Exercise the learning rule on CPU when server training dependencies are present."""

import io
import random
import unittest
from importlib.util import find_spec

if find_spec("torch") is not None:
    import torch

    from training.dqn import (
        CHANNELS, QNetwork, ReplayBuffer, make_network, mask_batch, optimize, state_batch,
    )
    from training.env import ACTIONS, SIDE
else:
    torch = None


@unittest.skipIf(torch is None, "requires the separate training environment")
class DoubleDQNTests(unittest.TestCase):
    def test_board_channels_reach_network_in_order(self):
        plane = SIDE * SIDE
        packed = bytes(plane) + bytes([1]) * plane + bytes(2 * plane)
        states = state_batch([packed, packed], torch.device("cpu"))
        self.assertEqual(tuple(states.shape), (2, CHANNELS, SIDE, SIDE))
        self.assertTrue(torch.all(states[:, 1] == 1).item())
        self.assertEqual(states.sum().item(), 2 * plane)
        self.assertEqual(tuple(QNetwork()(states).shape), (2, len(ACTIONS)))

    def test_v2_reads_four_directional_q_values_at_each_players_square(self):
        net = make_network(2)
        states = torch.zeros(2, CHANNELS, SIDE, SIDE)
        states[0, 3, 6, 3] = 1
        states[1, 3, 1, 9] = 1
        shape = (2, len(ACTIONS), SIDE, SIDE)
        # Distinct Q values let us check axis order independently of the weights.
        grid = torch.arange(states.numel(), dtype=torch.float32).reshape(shape)
        hook = net.head.register_forward_hook(lambda _layer, _inputs, _output: grid)
        try:
            values = net(states)
        finally:
            hook.remove()
        self.assertEqual(tuple(values.shape), (2, len(ACTIONS)))
        self.assertTrue(torch.equal(values[0], grid[0, :, 6, 3]))
        self.assertTrue(torch.equal(values[1], grid[1, :, 1, 9]))
        legal = mask_batch([(False, True, True, False), (True, False, False, True)], torch.device("cpu"))
        self.assertEqual(legal.dtype, torch.bool)
        self.assertEqual(values.masked_fill(~legal, -torch.inf).argmax(1).tolist(), [2, 3])

    def test_v1_loads_legacy_state_dict(self):
        original = QNetwork()
        walk = make_network(1)
        walk.load_state_dict(original.state_dict(), strict=True)
        states = torch.zeros(1, CHANNELS, SIDE, SIDE)
        self.assertTrue(torch.equal(walk(states), original(states)))

    def test_replay_resume_preserves_ring_overwrite_order(self):
        replay = ReplayBuffer(3)
        board = bytes(CHANNELS * SIDE * SIDE)
        for reward in range(5):
            replay.add(board, 0, float(reward), board, (True,) * 4, False)
        stream = io.BytesIO()
        torch.save(replay.state_dict(), stream)
        stream.seek(0)
        restored = ReplayBuffer(3)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        restored.add(board, 0, 5.0, board, (True,) * 4, False)
        self.assertEqual(sorted(row[2] for row in restored.sample(3, random.Random(1))), [3, 4, 5])

    def test_masked_online_selection_target_evaluation_and_terminal_backup(self):
        class FixedQ(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = torch.nn.Parameter(torch.tensor(values))

            def forward(self, states):
                return self.values.expand(len(states), -1)

        online = FixedQ([1.0, 3.0, 2.0, 8.0])
        target = FixedQ([2.0, 6.0, 9.0, 10.0])
        optimizer = torch.optim.SGD(online.parameters(), lr=0.01)
        replay = ReplayBuffer(3)
        board = bytes(CHANNELS * SIDE * SIDE)
        legal = (True, True, True, False)
        replay.add(board, 0, 0.0, board, legal, False)
        replay.add(board, 0, 0.0, board, legal, True)
        replay.add(board, 0, 0.0, board, (False,) * 4, False)
        loss = optimize(
            online, target, optimizer, replay, 3, 0.5, torch.device("cpu"), random.Random(1)
        )
        # Row 1's target is 0.5 * target[online_argmax_legal=1] = 3.
        # Other rows have zero bootstrap. Huber errors for prediction 1 are 1.5, 0.5, 0.5.
        self.assertAlmostEqual(loss.item(), 2.5 / 3, places=6)
        self.assertIsNone(target.values.grad)
