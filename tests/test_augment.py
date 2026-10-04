import torch

from flyracer.augment import random_shift


def test_zero_pad_is_identity():
    x = torch.randint(0, 255, (3, 4, 96, 96), dtype=torch.uint8)
    torch.testing.assert_close(random_shift(x, pad=0), x.float())


def test_shift_moves_whole_stack_together():
    x = torch.zeros(64, 4, 96, 96)
    x[:, :, 40:56, 40:56] = 255.0  # a bright square, same in every stacked frame
    y = random_shift(x, pad=4)
    assert y.shape == x.shape
    for i in range(64):
        rows, cols = torch.nonzero(y[i, 0] > 0, as_tuple=True)
        dy, dx = rows.min().item() - 40, cols.min().item() - 40
        assert abs(dy) <= 4 and abs(dx) <= 4
        torch.testing.assert_close(y[i, 1:], y[i, :1].expand(3, -1, -1))  # same shift per stack
    assert (y != x).any()  # something actually moved
