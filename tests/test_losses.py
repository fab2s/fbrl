"""Tests for fbrl/losses.py — all loss functions with synthetic tensors."""
import torch
import pytest

from fbrl.losses import (
    attention_content_loss,
    fixation_diversity_loss,
    fixation_edge_loss,
    fixation_hit_rate,
    two_phase_attention_loss,
    word_attention_loss,
)


@pytest.fixture
def bright_image():
    """Image with content everywhere (all ones)."""
    return torch.ones(4, 1, 128, 128)


@pytest.fixture
def dark_image():
    """Image with no content (all zeros)."""
    return torch.zeros(4, 1, 128, 128)


class TestAttentionContentLoss:
    def test_scalar_finite(self, bright_image):
        locs = [torch.zeros(4, 2)] + [torch.randn(4, 2) * 0.5 for _ in range(5)]
        loss = attention_content_loss(bright_image, locs)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_negative_guide(self, bright_image):
        """Loss should be <= 0 since it's negated guide value."""
        locs = [torch.zeros(4, 2)] + [torch.zeros(4, 2) for _ in range(5)]
        loss = attention_content_loss(bright_image, locs)
        assert loss.item() <= 0.0

    def test_differentiable(self, bright_image):
        locs = [torch.zeros(4, 2)] + [torch.randn(4, 2, requires_grad=True) for _ in range(3)]
        loss = attention_content_loss(bright_image, locs)
        loss.backward()
        assert locs[1].grad is not None


class TestFixationDiversityLoss:
    def test_scalar_nonneg(self):
        locs = [torch.zeros(4, 2)] + [torch.randn(4, 2) for _ in range(5)]
        loss = fixation_diversity_loss(locs)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_close_fixations_high(self):
        """Fixations at the same point should give high repulsion."""
        same = torch.zeros(4, 2)
        locs = [torch.zeros(4, 2)] + [same for _ in range(5)]
        loss_close = fixation_diversity_loss(locs)

        far = [torch.zeros(4, 2)] + [torch.tensor([[i, 0.0]]).expand(4, -1)
                                       for i in range(-2, 3)]
        loss_far = fixation_diversity_loss(far)
        assert loss_close.item() > loss_far.item()

    def test_vy_scaling(self):
        """Higher vy amplifies vertical distances, reducing repulsion for vertically-separated points."""
        locs = [torch.zeros(4, 2)]
        locs += [torch.tensor([[0.0, 0.05]]).expand(4, -1),
                 torch.tensor([[0.0, -0.05]]).expand(4, -1)]
        loss_vy1 = fixation_diversity_loss(locs, vy=1.0)
        loss_vy3 = fixation_diversity_loss(locs, vy=3.0)
        # vy > 1 scales vertical diff, making points appear further apart -> less repulsion
        assert loss_vy3.item() < loss_vy1.item()

    def test_single_fixation_zero(self):
        """Single fixation -> no pairs -> diversity = 0."""
        locs = [torch.zeros(4, 2), torch.randn(4, 2)]
        loss = fixation_diversity_loss(locs)
        assert loss.item() == 0.0


class TestFixationEdgeLoss:
    def test_center_max(self):
        """Fixations at center x=0 should give max loss (1.0)."""
        locs = [torch.zeros(4, 2), torch.zeros(4, 2)]
        loss = fixation_edge_loss(locs)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_edges_low(self):
        """Fixations at edges x=±1 should give ~0 loss."""
        locs = [torch.zeros(4, 2)]
        edge = torch.tensor([[1.0, 0.0]]).expand(4, -1)
        locs.append(edge)
        loss = fixation_edge_loss(locs)
        assert loss.item() < 0.01


class TestFixationHitRate:
    def test_bright_high(self, bright_image):
        locs = [torch.zeros(4, 2)] + [torch.randn(4, 2) * 0.5 for _ in range(5)]
        hit_rate, intensity = fixation_hit_rate(bright_image, locs)
        assert hit_rate >= 0.9

    def test_dark_low(self, dark_image):
        locs = [torch.zeros(4, 2)] + [torch.randn(4, 2) * 0.5 for _ in range(5)]
        hit_rate, intensity = fixation_hit_rate(dark_image, locs)
        assert hit_rate == 0.0


class TestTwoPhaseAttentionLoss:
    def test_returns_two_scalars(self, bright_image):
        locs = [torch.zeros(4, 2)] + [torch.randn(4, 2) for _ in range(9)]
        scan_loss, read_loss = two_phase_attention_loss(bright_image, locs, n_scan=3)
        assert scan_loss.shape == ()
        assert read_loss.shape == ()
        assert torch.isfinite(scan_loss)
        assert torch.isfinite(read_loss)


class TestWordAttentionLoss:
    def test_returns_two_scalars(self, bright_image):
        scan_locs = [torch.randn(4, 2) for _ in range(4)]
        read_locs = [torch.randn(4, 2) for _ in range(8)]
        scan_loss, read_loss = word_attention_loss(bright_image, scan_locs, read_locs)
        assert scan_loss.shape == ()
        assert read_loss.shape == ()
        assert torch.isfinite(scan_loss)
        assert torch.isfinite(read_loss)
