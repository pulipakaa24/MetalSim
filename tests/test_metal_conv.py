"""Custom Metal conv-gradient kernels (metalsim/learn/metal_conv.py) match PyTorch's MPS gradients in fp32 (1e-4)."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _rel(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)


def _ref(gy, x, w, s, mask):
    return torch.ops.aten.convolution_backward(gy, x, w, [w.shape[0]], [s, s], [0, 0], [1, 1], False, [0, 0], 1, mask)


@pytest.mark.parametrize("n,u8", [(2048, True), (37, False)])
def test_conv1_weight_grad(n, u8):
    from metalsim.learn.metal_conv import conv1_weight_grad
    g = torch.Generator().manual_seed(n)
    x = (torch.randint(0, 256, (n, 3, 100, 100), generator=g).float() if u8 else torch.randn(n, 3, 100, 100, generator=g)).to("mps")
    w = (torch.randn(32, 3, 8, 8, generator=g) * 0.05).to("mps")
    gy = torch.randn(n, 32, 24, 24, generator=g).to("mps")
    assert _rel(conv1_weight_grad(x, gy), _ref(gy, x, w, 4, [False, True, False])[1]) <= 1e-4


@pytest.mark.parametrize("n", [2048, 37])
def test_conv2_input_grad(n):
    from metalsim.learn.metal_conv import conv2_input_grad
    g = torch.Generator().manual_seed(n)
    x = torch.randn(n, 32, 24, 24, generator=g).to("mps")
    w = (torch.randn(64, 32, 4, 4, generator=g) * 0.05).to("mps")
    gy = torch.randn(n, 64, 11, 11, generator=g).to("mps")
    assert _rel(conv2_input_grad(gy, w), _ref(gy, x, w, 2, [True, False, False])[0]) <= 1e-4


def test_functions_all_grads_match():
    """The autograd Functions reproduce every gradient (x, w, b) of the two convs, eager and under torch.compile."""
    from metalsim.learn import metal_conv as mc
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(256, 3, 100, 100, generator=g).to("mps").requires_grad_(True)
    w1 = (torch.randn(32, 3, 8, 8, generator=g) * 0.05).to("mps").requires_grad_(True)
    b1 = torch.randn(32, generator=g).to("mps").requires_grad_(True)
    w2 = (torch.randn(64, 32, 4, 4, generator=g) * 0.05).to("mps").requires_grad_(True)
    b2 = torch.randn(64, generator=g).to("mps").requires_grad_(True)
    t = torch.randn(256, 64, 11, 11, generator=g).to("mps")

    def f(c1, c2):
        return (c2(F.relu(c1(x1, w1, b1, 4)), w2, b2, 2) * t).sum()

    ref = torch.autograd.grad(f(lambda *a: F.conv2d(*a[:3], stride=a[3]), lambda *a: F.conv2d(*a[:3], stride=a[3])),
                              [x1, w1, b1, w2, b2])
    for fn in (f, torch.compile(f)):
        got = torch.autograd.grad(fn(mc.conv1, mc.conv2), [x1, w1, b1, w2, b2])
        for r, o in zip(ref, got):
            assert _rel(o, r) <= 1e-4
