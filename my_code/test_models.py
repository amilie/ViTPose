"""
test_models.py — offline unit tests for both model architectures.

Does NOT require the dataset to be present. All tests use synthetic random
tensors so they can be run anywhere (CI, laptop, no data attached).

Run with:
    python3 test_models.py           # prints pass/fail for every check
    python3 -m pytest test_models.py -v   # if pytest is installed
"""

import time
import torch
import torch.nn as nn

from model import SimpleRadarPoseCNN, RadarPoseResTransformer

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BATCH = 4
NUM_KP = 17
OUT_SHAPE = (BATCH, NUM_KP, 2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(batch: int = BATCH) -> torch.Tensor:
    """Synthetic radar input in [0, 1]."""
    return torch.rand(batch, 2, 256, 128, device=DEVICE)


# ---------------------------------------------------------------------------
# Helper checks — called by both the pytest classes and the standalone runner
# ---------------------------------------------------------------------------

def _check_output_shape(model: nn.Module, name: str) -> None:
    x = make_batch()
    with torch.no_grad():
        out = model(x)
    assert out.shape == OUT_SHAPE, f"[{name}] Expected {OUT_SHAPE}, got {out.shape}"


def _check_no_nan_inf(model: nn.Module, name: str) -> None:
    x = make_batch()
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), f"[{name}] Output contains NaN"
    assert not torch.isinf(out).any(), f"[{name}] Output contains Inf"


def _check_gradient_flows(model: nn.Module, name: str) -> None:
    """backward() must produce non-zero gradients for every trainable parameter."""
    model.train()
    x = make_batch()
    target = torch.rand(*OUT_SHAPE, device=DEVICE)
    out = model(x)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    dead = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)
    ]
    model.eval()
    assert not dead, f"[{name}] Dead gradients in: {dead[:5]}"


def _check_batch_size_1(model: nn.Module, name: str) -> None:
    """Single-sample inference must not crash (exercises BatchNorm in eval mode)."""
    x = make_batch(batch=1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, NUM_KP, 2), f"[{name}] Wrong shape for batch=1: {out.shape}"


def _check_output_varies(model: nn.Module, name: str) -> None:
    x1, x2 = make_batch(), torch.rand_like(make_batch())
    with torch.no_grad():
        o1, o2 = model(x1), model(x2)
    assert not torch.allclose(o1, o2), f"[{name}] Identical output for different inputs"


def _check_deterministic_eval(model: nn.Module, name: str) -> None:
    """Same input in eval mode must always give the same output."""
    x = make_batch()
    with torch.no_grad():
        o1, o2 = model(x), model(x)
    assert torch.allclose(o1, o2), f"[{name}] eval() output is not deterministic"


def _check_custom_keypoints(cls, num_kp: int, name: str) -> None:
    model = cls(num_keypoints=num_kp).to(DEVICE).eval()
    x = make_batch()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH, num_kp, 2), f"[{name}] Wrong shape for {num_kp} kp: {out.shape}"


# ---------------------------------------------------------------------------
# pytest classes — only defined when pytest is importable
# ---------------------------------------------------------------------------

try:
    import pytest

    @pytest.fixture(scope="module")
    def cnn_model():
        m = SimpleRadarPoseCNN(num_keypoints=NUM_KP).to(DEVICE)
        m.eval()
        return m

    @pytest.fixture(scope="module")
    def transformer_model():
        m = RadarPoseResTransformer(num_keypoints=NUM_KP, d_model=128, nhead=4, num_transformer_layers=2).to(DEVICE)
        m.eval()
        return m

    class TestSimpleRadarPoseCNN:
        def test_output_shape(self, cnn_model):         _check_output_shape(cnn_model, "CNN")
        def test_no_nan_inf(self, cnn_model):           _check_no_nan_inf(cnn_model, "CNN")
        def test_gradient_flows(self, cnn_model):       _check_gradient_flows(cnn_model, "CNN")
        def test_batch_size_1(self, cnn_model):         _check_batch_size_1(cnn_model, "CNN")
        def test_output_varies(self, cnn_model):        _check_output_varies(cnn_model, "CNN")
        def test_deterministic_eval(self, cnn_model):   _check_deterministic_eval(cnn_model, "CNN")
        def test_param_count(self, cnn_model):
            n = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
            assert 10_000 < n < 50_000_000, f"Suspicious param count: {n}"
        def test_custom_keypoints(self):
            _check_custom_keypoints(SimpleRadarPoseCNN, 13, "CNN")

    class TestRadarPoseResTransformer:
        def test_output_shape(self, transformer_model):         _check_output_shape(transformer_model, "RTR")
        def test_no_nan_inf(self, transformer_model):           _check_no_nan_inf(transformer_model, "RTR")
        def test_gradient_flows(self, transformer_model):       _check_gradient_flows(transformer_model, "RTR")
        def test_batch_size_1(self, transformer_model):         _check_batch_size_1(transformer_model, "RTR")
        def test_output_varies(self, transformer_model):        _check_output_varies(transformer_model, "RTR")
        def test_deterministic_eval(self, transformer_model):   _check_deterministic_eval(transformer_model, "RTR")
        def test_param_count(self, transformer_model):
            n = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
            assert 10_000 < n < 100_000_000, f"Suspicious param count: {n}"
        def test_custom_keypoints(self):
            _check_custom_keypoints(RadarPoseResTransformer, 13, "RTR")
        def test_custom_transformer_depth(self):
            for layers in (1, 3, 4):
                m = RadarPoseResTransformer(num_transformer_layers=layers).to(DEVICE).eval()
                with torch.no_grad():
                    out = m(make_batch())
                assert out.shape == OUT_SHAPE, f"Wrong shape with {layers} layers"

    class TestComparison:
        def test_same_output_shape(self, cnn_model, transformer_model):
            x = make_batch()
            with torch.no_grad():
                assert cnn_model(x).shape == transformer_model(x).shape == OUT_SHAPE

        def test_inference_speed(self, cnn_model, transformer_model):
            x, N = make_batch(), 50
            with torch.no_grad():
                cnn_model(x); transformer_model(x)  # warm up
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(N): cnn_model(x)
            cnn_t = time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(N): transformer_model(x)
            rtr_t = time.perf_counter() - t0
            print(f"\n  CNN {cnn_t/N*1000:.1f}ms/pass  RTR {rtr_t/N*1000:.1f}ms/pass")
            assert cnn_t < 60 and rtr_t < 60

except ImportError:
    pass  # pytest not installed — use the standalone runner below


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    print(f"Device: {DEVICE}\n")

    cnn = SimpleRadarPoseCNN(num_keypoints=NUM_KP).to(DEVICE).eval()
    rtr = RadarPoseResTransformer(num_keypoints=NUM_KP, d_model=128, nhead=4, num_transformer_layers=2).to(DEVICE).eval()

    tests = [
        ("SimpleRadarPoseCNN   output shape",       lambda: _check_output_shape(cnn, "CNN")),
        ("SimpleRadarPoseCNN   no NaN/Inf",         lambda: _check_no_nan_inf(cnn, "CNN")),
        ("SimpleRadarPoseCNN   gradient flows",     lambda: _check_gradient_flows(cnn, "CNN")),
        ("SimpleRadarPoseCNN   batch size 1",       lambda: _check_batch_size_1(cnn, "CNN")),
        ("SimpleRadarPoseCNN   output varies",      lambda: _check_output_varies(cnn, "CNN")),
        ("SimpleRadarPoseCNN   deterministic eval", lambda: _check_deterministic_eval(cnn, "CNN")),
        ("SimpleRadarPoseCNN   custom kp=13",       lambda: _check_custom_keypoints(SimpleRadarPoseCNN, 13, "CNN")),
        ("RadarPoseResTransf   output shape",       lambda: _check_output_shape(rtr, "RTR")),
        ("RadarPoseResTransf   no NaN/Inf",         lambda: _check_no_nan_inf(rtr, "RTR")),
        ("RadarPoseResTransf   gradient flows",     lambda: _check_gradient_flows(rtr, "RTR")),
        ("RadarPoseResTransf   batch size 1",       lambda: _check_batch_size_1(rtr, "RTR")),
        ("RadarPoseResTransf   output varies",      lambda: _check_output_varies(rtr, "RTR")),
        ("RadarPoseResTransf   deterministic eval", lambda: _check_deterministic_eval(rtr, "RTR")),
        ("RadarPoseResTransf   custom kp=13",       lambda: _check_custom_keypoints(RadarPoseResTransformer, 13, "RTR")),
    ]

    passed = failed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"  PASS  {label}")
            passed += 1
        except Exception:
            print(f"  FAIL  {label}")
            traceback.print_exc()
            failed += 1

    # Parameter counts
    cnn.eval(); rtr.eval()
    cnn_p = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
    rtr_p = sum(p.numel() for p in rtr.parameters() if p.requires_grad)
    print(f"\n  SimpleRadarPoseCNN      params: {cnn_p:,}")
    print(f"  RadarPoseResTransformer params: {rtr_p:,}")

    # Speed comparison
    x, N = make_batch(), 50
    with torch.no_grad():
        cnn(x); rtr(x)  # warm up

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N): cnn(x)
    cnn_ms = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N): rtr(x)
    rtr_ms = (time.perf_counter() - t0) / N * 1000

    print(f"\n  Inference speed (batch={BATCH}, {N} passes):")
    print(f"    SimpleRadarPoseCNN     : {cnn_ms:.1f} ms/pass")
    print(f"    RadarPoseResTransformer: {rtr_ms:.1f} ms/pass")

    print(f"\n{'='*52}")
    print(f"  {passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
