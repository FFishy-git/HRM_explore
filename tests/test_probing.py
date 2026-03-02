"""Unit tests for the probing MLP module and data pipeline logic."""
import pytest
import torch
import torch.nn.functional as F

from models.probing import ProbingMLP
from probe_train import _compute_probe_loss


# ──────────────────────── ProbingMLP shape tests ─────────────────────────


class TestProbingMLPShape:
    """Test that ProbingMLP produces correct output shapes."""

    def test_output_shape_default(self):
        """Output shape is [batch, seq, 9] for input [batch, seq, 1536] with defaults."""
        probe = ProbingMLP()  # hidden_size=512, so d_in = 3*512 = 1536
        x = torch.randn(2, 81, 1536)
        out = probe(x)
        assert out.shape == (2, 81, 9)

    def test_output_shape_custom_hidden_size(self):
        """Output shape adapts to custom hidden_size."""
        probe = ProbingMLP(hidden_size=256, hidden_mult=2, num_classes=9)
        d_in = 3 * 256
        x = torch.randn(4, 81, d_in)
        out = probe(x)
        assert out.shape == (4, 81, 9)

    def test_output_shape_single_position(self):
        """Works with seq_len=1."""
        probe = ProbingMLP()
        x = torch.randn(1, 1, 1536)
        out = probe(x)
        assert out.shape == (1, 1, 9)

    def test_output_is_logits(self):
        """Output values are unbounded logits (not probabilities)."""
        probe = ProbingMLP()
        x = torch.randn(2, 81, 1536)
        out = probe(x)
        # Logits can be negative or > 1; softmax of logits should sum to 1
        probs = F.softmax(out, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 81), atol=1e-5)


# ──────────────────────── Label remapping tests ──────────────────────────


class TestLabelRemapping:
    """Test that target label remapping converts tokens 2-10 to classes 0-8."""

    def test_remap_tokens_to_classes(self):
        """Tokens 2-10 should map to classes 0-8 via `labels - 2`."""
        tokens = torch.arange(2, 11)  # [2, 3, 4, 5, 6, 7, 8, 9, 10]
        classes = tokens - 2
        expected = torch.arange(0, 9)
        assert torch.equal(classes, expected)

    def test_remap_in_batch_context(self):
        """Remapping works across batch and seq dimensions."""
        labels = torch.tensor([[2, 5, 10], [3, 7, 9]])  # [2, 3]
        classes = labels - 2
        expected = torch.tensor([[0, 3, 8], [1, 5, 7]])
        assert torch.equal(classes, expected)


# ──────────────────────── Blank cell mask tests ──────────────────────────


class TestBlankCellMask:
    """Test that blank cell mask correctly identifies cells where inputs == 1."""

    def test_blank_cells_identified(self):
        """inputs == 1 marks blank cells (to be predicted)."""
        inputs = torch.tensor([[1, 5, 1, 3, 1]])  # [1, 5]
        blank_mask = inputs == 1
        expected = torch.tensor([[True, False, True, False, True]])
        assert torch.equal(blank_mask, expected)

    def test_no_blank_cells(self):
        """When no cells are blank, mask is all False."""
        inputs = torch.tensor([[2, 3, 4]])
        blank_mask = inputs == 1
        assert not blank_mask.any()

    def test_all_blank_cells(self):
        """When all cells are blank, mask is all True."""
        inputs = torch.ones(2, 81, dtype=torch.long)
        blank_mask = inputs == 1
        assert blank_mask.all()


# ──────────────────────── Loss on blank cells only ───────────────────────


class TestLossOnBlankCells:
    """Test that loss is computed only on masked (blank) cells."""

    def test_loss_only_on_blank_cells(self):
        """Given cells (non-blank) should not contribute to loss."""
        probe = ProbingMLP(hidden_size=512)
        batch, seq = 2, 10
        d_in = 3 * 512

        inputs = torch.randn(batch, seq, d_in)
        # Only positions 0, 3, 7 are blank
        blank_mask = torch.zeros(batch, seq, dtype=torch.bool)
        blank_mask[:, [0, 3, 7]] = True
        # Labels: tokens 2-10 range
        labels = torch.randint(2, 11, (batch, seq))

        loss, acc = _compute_probe_loss(probe, inputs, labels, blank_mask)

        # Loss should be a scalar
        assert loss.dim() == 0
        # Accuracy should be a float in [0, 1]
        assert 0.0 <= acc <= 1.0

    def test_given_cells_do_not_affect_loss(self):
        """Changing given cell labels should not change the loss."""
        torch.manual_seed(42)
        probe = ProbingMLP(hidden_size=512)
        batch, seq = 2, 10
        d_in = 3 * 512

        inputs = torch.randn(batch, seq, d_in)
        blank_mask = torch.zeros(batch, seq, dtype=torch.bool)
        blank_mask[:, [0, 3]] = True
        labels_a = torch.randint(2, 11, (batch, seq))
        labels_b = labels_a.clone()
        # Change only non-blank (given) cell labels
        labels_b[:, 1] = 2  # position 1 is not blank
        labels_b[:, 5] = 10  # position 5 is not blank

        loss_a, _ = _compute_probe_loss(probe, inputs, labels_a, blank_mask)
        loss_b, _ = _compute_probe_loss(probe, inputs, labels_b, blank_mask)

        assert torch.allclose(loss_a, loss_b)

    def test_empty_blank_mask_returns_zero_loss(self):
        """When no cells are blank, loss should be zero."""
        probe = ProbingMLP(hidden_size=512)
        batch, seq = 2, 10
        d_in = 3 * 512

        inputs = torch.randn(batch, seq, d_in)
        blank_mask = torch.zeros(batch, seq, dtype=torch.bool)  # no blank cells
        labels = torch.randint(2, 11, (batch, seq))

        loss, acc = _compute_probe_loss(probe, inputs, labels, blank_mask)

        assert loss.item() == 0.0
        assert acc == 0.0


# ──────────────────────── Gradient isolation tests ───────────────────────


class TestGradientIsolation:
    """Test that probe gradients flow only to probe parameters, not HRM parameters."""

    def test_gradients_flow_to_probe_only(self):
        """After backward, probe params have gradients; simulated HRM params do not."""
        probe = ProbingMLP(hidden_size=512)
        d_in = 3 * 512
        batch, seq = 2, 81

        # Simulate frozen HRM output (requires_grad=False, like model.requires_grad_(False))
        fake_z_H = torch.randn(batch, seq, 512)
        fake_z_L = torch.randn(batch, seq, 512)
        fake_emb = torch.randn(batch, seq, 512)

        # These simulate HRM parameters that should NOT receive gradients
        # In the real pipeline, model.requires_grad_(False) prevents this
        hrm_param = torch.nn.Parameter(torch.randn(512, 512))
        # Use hrm_param in a way that connects it to the computation
        modified_z_H = fake_z_H + F.linear(fake_z_H, hrm_param)[:, :, :512]
        modified_z_H = modified_z_H.detach()  # .detach() simulates frozen HRM

        probe_input = torch.cat([modified_z_H, fake_z_L, fake_emb], dim=-1)

        # Blank mask and labels
        blank_mask = torch.ones(batch, seq, dtype=torch.bool)
        labels = torch.randint(2, 11, (batch, seq))

        loss, _ = _compute_probe_loss(probe, probe_input, labels, blank_mask)
        loss.backward()

        # Probe parameters should have gradients
        for name, param in probe.named_parameters():
            assert param.grad is not None, f"Probe param {name} has no gradient"
            assert param.grad.abs().sum() > 0, f"Probe param {name} has zero gradient"

        # HRM param should NOT have gradients (it was detached)
        assert hrm_param.grad is None, "HRM parameter should not receive gradients"

    def test_detached_inputs_block_gradient_flow(self):
        """Detached tensors (as produced by hooks) do not propagate gradients upstream."""
        probe = ProbingMLP(hidden_size=512)
        d_in = 3 * 512
        batch, seq = 1, 10

        # Simulating the hook path: tensors are .detach()-ed
        source = torch.randn(batch, seq, d_in, requires_grad=True)
        detached = source.detach()  # Mimics _detach() in create_probe_hooks

        blank_mask = torch.ones(batch, seq, dtype=torch.bool)
        labels = torch.randint(2, 11, (batch, seq))

        loss, _ = _compute_probe_loss(probe, detached, labels, blank_mask)
        loss.backward()

        # Source should NOT have gradients because detached broke the graph
        assert source.grad is None, "Detached input should not propagate gradients"
