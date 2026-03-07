"""Tests for fbrl/model.py — forward pass shapes with small dummy tensors."""
import torch
import pytest

from fbrl.model import (
    VisionModel,
    MotorVisionModel,
    BigramVisionModel,
    WordVisionModel,
    CrossAttentionReadout,
)

B = 2  # small batch for speed


class TestVisionModel:
    def test_forward_shapes(self):
        model = VisionModel(n_classes=26, latent_dim=128, n_glimpses=3, patch_size=8)
        img = torch.randn(B, 1, 128, 128)
        case_label = torch.zeros(B, 1)
        recon, letter, case, locs, latent, scan_cl = model(img, case_label)
        assert recon.shape == (B, 1, 128, 128)
        assert letter.shape == (B, 26)
        assert case.shape == (B, 2)
        assert latent.shape == (B, 128)
        assert torch.isfinite(recon).all()
        assert torch.isfinite(letter).all()

    def test_with_scan(self):
        model = VisionModel(
            n_classes=26, latent_dim=128, n_glimpses=3, patch_size=8,
            n_scan_glimpses=2, scan_patch_size=(8, 12),
        )
        img = torch.randn(B, 1, 128, 128)
        case_label = torch.zeros(B, 1)
        recon, letter, case, locs, latent, scan_cl = model(img, case_label)
        assert recon.shape == (B, 1, 128, 128)
        # scan_content_logits should have one per scan glimpse
        assert len(scan_cl) == 2
        # More locations: 1 start + 2 scan + 3 read = 6
        assert len(locs) == 6

    def test_recode(self):
        model = VisionModel(n_classes=26, latent_dim=128, n_glimpses=3, patch_size=8)
        img = torch.randn(B, 1, 128, 128)
        target_case = torch.ones(B, 1)
        recon, locs = model.recode(img, target_case)
        assert recon.shape == (B, 1, 128, 128)
        assert len(locs) > 0


class TestMotorVisionModel:
    def test_forward_and_motor(self):
        model = MotorVisionModel(
            n_classes=26, latent_dim=128, n_glimpses=3, patch_size=8,
            n_trajectory_points=16, render_sigma=1.5,
        )
        img = torch.randn(B, 1, 128, 128)
        case_label = torch.zeros(B, 1)
        recon, letter, case, locs, latent, scan_cl = model(img, case_label)
        assert recon.shape == (B, 1, 128, 128)

        traj, rendered = model.motor_forward(latent.detach())
        assert traj.shape == (B, 16, 3)
        assert rendered.shape == (B, 1, 128, 128)
        assert torch.isfinite(traj).all()
        assert torch.isfinite(rendered).all()


class TestBigramVisionModel:
    def test_forward_shapes(self):
        model = BigramVisionModel(
            n_classes=26, latent_dim=128,
            n_scan_glimpses=2, n_read_glimpses=3,
            scan_patch_size=(8, 12), read_patch_size=8,
        )
        img = torch.randn(B, 1, 128, 128)
        recon, logits_list, locs, readout = model(img)
        assert recon.shape == (B, 1, 128, 128)
        assert len(logits_list) == 2
        assert logits_list[0].shape == (B, 26)
        assert logits_list[1].shape == (B, 26)
        assert readout.shape == (B, 2, 128)
        assert torch.isfinite(recon).all()


class TestWordVisionModel:
    def test_forward_shapes(self):
        model = WordVisionModel(
            n_classes=26, latent_dim=128,
            n_scan_glimpses=3, n_read_glimpses=4,
            scan_patch_size=(8, 12), read_patch_size=8,
        )
        img = torch.randn(B, 1, 128, 256)
        recon, logits_list, locs, readout, scan_cl, group_bounds, phase_tags = model(img)
        assert recon.shape == (B, 1, 128, 256)
        assert len(logits_list) == 4
        for lg in logits_list:
            assert lg.shape == (B, 26)
        assert readout.shape == (B, 4, 128)
        assert torch.isfinite(recon).all()


class TestCrossAttentionReadout:
    def test_output_shape(self):
        readout = CrossAttentionReadout(latent_dim=128, n_positions=3)
        states = torch.randn(B, 6, 128)
        out = readout(states)
        assert out.shape == (B, 3, 128)
        assert torch.isfinite(out).all()
