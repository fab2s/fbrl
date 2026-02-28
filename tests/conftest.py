"""Shared fixtures for FBRL test suite."""
import pytest
import torch


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def B():
    return 4


@pytest.fixture
def small_image(B):
    return torch.randn(B, 1, 128, 128)


@pytest.fixture
def locations(B):
    return [torch.randn(B, 2) for _ in range(5)]
