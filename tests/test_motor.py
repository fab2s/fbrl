"""Tests for fbrl/motor.py — trajectory math, renderer, decoder."""
import torch
import pytest

from fbrl.motor import (
    _flatten_cubic,
    _resample_trajectory,
    _fallback_trajectory,
    resolve_font_path,
    batch_gt_trajectories,
    soft_render,
    MotorTraceDecoder,
)


class TestFlattenCubic:
    def test_length(self):
        pts = _flatten_cubic((0, 0), (0.33, 1), (0.66, 1), (1, 0), n_segments=8)
        assert len(pts) == 8

    def test_endpoints(self):
        p0, p3 = (0.0, 0.0), (1.0, 0.0)
        pts = _flatten_cubic(p0, (0.33, 1.0), (0.66, 1.0), p3, n_segments=4)
        # Last point should be near p3
        assert abs(pts[-1][0] - p3[0]) < 1e-6
        assert abs(pts[-1][1] - p3[1]) < 1e-6


class TestResampleTrajectory:
    def test_output_length(self):
        points = [(0, 0), (1, 0), (2, 0), (3, 0)]
        pen = [0.0, 1.0, 1.0, 1.0]
        resampled, pen_out = _resample_trajectory(points, pen, n_points=16)
        assert len(resampled) == 16
        assert len(pen_out) == 16

    def test_pen_states_binary(self):
        points = [(0, 0), (1, 0), (2, 1)]
        pen = [0.0, 1.0, 1.0]
        _, pen_out = _resample_trajectory(points, pen, n_points=10)
        assert all(p in (0.0, 1.0) for p in pen_out)

    def test_single_point(self):
        points = [(0.5, 0.5)]
        pen = [1.0]
        resampled, pen_out = _resample_trajectory(points, pen, n_points=8)
        assert len(resampled) == 8
        assert all(p == (0.5, 0.5) for p in resampled)


class TestFallbackTrajectory:
    def test_shape(self):
        t = _fallback_trajectory(32)
        assert t.shape == (32, 3)

    def test_all_zeros(self):
        t = _fallback_trajectory(16)
        assert (t == 0).all()


class TestResolveFontPath:
    def test_known_font(self):
        path = resolve_font_path('dejavu-sans')
        assert 'DejaVuSans' in path

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown font'):
            resolve_font_path('nonexistent-font-xyz')


class TestBatchGtTrajectories:
    def test_shape_and_case(self):
        traj_data = {
            'A': torch.randn(32, 3),
            'a': torch.randn(32, 3),
            'B': torch.randn(32, 3),
            'b': torch.randn(32, 3),
        }
        letters = ['A', 'A', 'B', 'B']
        cases = ['upper', 'lower', 'upper', 'lower']
        result = batch_gt_trajectories(letters, cases, traj_data, torch.device('cpu'))
        assert result.shape == (4, 32, 3)
        # Verify case mapping: upper 'A' should use traj_data['A']
        assert torch.allclose(result[0], traj_data['A'])
        # lower 'A' should use traj_data['a']
        assert torch.allclose(result[1], traj_data['a'])


class TestSoftRender:
    def test_shape(self):
        traj = torch.randn(4, 16, 3)
        canvas = soft_render(traj, height=64, width=64)
        assert canvas.shape == (4, 1, 64, 64)

    def test_range(self):
        traj = torch.randn(4, 16, 3)
        canvas = soft_render(traj, height=64, width=64)
        assert canvas.min() >= 0.0
        assert canvas.max() <= 1.0

    def test_no_nan(self):
        traj = torch.randn(4, 16, 3)
        canvas = soft_render(traj, height=64, width=64)
        assert torch.isfinite(canvas).all()

    def test_gradient_flows(self):
        traj = torch.randn(4, 8, 3, requires_grad=True)
        canvas = soft_render(traj, height=32, width=32)
        canvas.sum().backward()
        assert traj.grad is not None
        assert torch.isfinite(traj.grad).all()


class TestMotorTraceDecoder:
    def test_output_shape(self):
        dec = MotorTraceDecoder(latent_dim=256, hidden_dim=128, n_points=32)
        latent = torch.randn(4, 256)
        out = dec(latent)
        assert out.shape == (4, 32, 3)

    def test_xy_range(self):
        """xy should be in [-1, 1] from tanh."""
        dec = MotorTraceDecoder(latent_dim=256, hidden_dim=128, n_points=16)
        latent = torch.randn(4, 256)
        out = dec(latent)
        xy = out[:, :, :2]
        assert xy.min() >= -1.0
        assert xy.max() <= 1.0
