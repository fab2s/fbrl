"""Tests for fbrl/training_utils.py — LossTracker, format_eta, checkpoints."""
import os
import tempfile
import torch
import pytest

from fbrl.training_utils import LossTracker, format_eta, save_checkpoint, load_checkpoint


class TestLossTracker:
    def test_basic_flow(self):
        tracker = LossTracker(['loss_a', 'loss_b'])
        tracker.reset_epoch()
        tracker.update(loss_a=1.0, loss_b=2.0)
        tracker.update(loss_a=3.0, loss_b=4.0)
        avgs = tracker.end_epoch()
        assert abs(avgs['loss_a'] - 2.0) < 1e-6
        assert abs(avgs['loss_b'] - 3.0) < 1e-6

    def test_multiple_epochs(self):
        tracker = LossTracker(['x'])
        for val in [1.0, 5.0]:
            tracker.reset_epoch()
            tracker.update(x=val)
            tracker.end_epoch()
        assert len(tracker.history['x']) == 2
        assert abs(tracker.history['x'][0] - 1.0) < 1e-6
        assert abs(tracker.history['x'][1] - 5.0) < 1e-6

    def test_torch_tensor_values(self):
        tracker = LossTracker(['t'])
        tracker.reset_epoch()
        tracker.update(t=torch.tensor(3.14))
        avgs = tracker.end_epoch()
        assert abs(avgs['t'] - 3.14) < 1e-5

    def test_unknown_keys_ignored(self):
        tracker = LossTracker(['known'])
        tracker.reset_epoch()
        tracker.update(known=1.0, unknown=999.0)
        avgs = tracker.end_epoch()
        assert 'unknown' not in avgs
        assert abs(avgs['known'] - 1.0) < 1e-6

    def test_history_roundtrip(self):
        tracker = LossTracker(['a', 'b'])
        tracker.reset_epoch()
        tracker.update(a=10.0, b=20.0)
        tracker.end_epoch()

        hist = tracker.get_history_dict()
        tracker2 = LossTracker(['a', 'b'])
        tracker2.restore_history(hist)
        assert tracker2.history['a'] == tracker.history['a']
        assert tracker2.history['b'] == tracker.history['b']


class TestFormatEta:
    def test_known_input(self):
        result = format_eta(elapsed=60.0, done=10, remaining_ep=20)
        assert result == '2m00s'

    def test_done_zero(self):
        assert format_eta(elapsed=100.0, done=0, remaining_ep=50) == '?'

    def test_fractional(self):
        result = format_eta(elapsed=90.0, done=3, remaining_ep=1)
        assert result == '0m30s'


class TestCheckpointRoundtrip:
    def test_save_load(self):
        model = torch.nn.Linear(10, 5)
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, epoch=7, path=path)
            ckpt = load_checkpoint(path, device=torch.device('cpu'))
            assert ckpt['epoch'] == 7
            assert 'model' in ckpt
            # Verify state dict keys
            for key in model.state_dict():
                assert key in ckpt['model']
        finally:
            os.unlink(path)
