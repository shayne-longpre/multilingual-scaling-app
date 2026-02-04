"""
Test suite for verifying consistent function signatures across all scaling law classes.

These tests ensure that torch_loss and numpy_loss methods have consistent signatures
matching the updated ChinchillaScalingLaw implementation.
"""

import pytest
import numpy as np
import torch

from src.scaling_law_classes.scaling_law import ScalingLaw
from src.scaling_law_classes.chinchilla_scaling_law import ChinchillaScalingLaw
from src.scaling_law_classes.basic_scaling_law import BasicScalingLaw
from src.scaling_law_classes.data_constrained_scaling_law import DataConstrainedScalingLaw
from src.scaling_law_classes.general_scaling_law import GeneralScalingLaw


class TestImports:
    """Test that all scaling law classes can be imported."""

    def test_import_scaling_law(self):
        assert ScalingLaw is not None

    def test_import_chinchilla_scaling_law(self):
        assert ChinchillaScalingLaw is not None

    def test_import_basic_scaling_law(self):
        assert BasicScalingLaw is not None

    def test_import_data_constrained_scaling_law(self):
        assert DataConstrainedScalingLaw is not None

    def test_import_general_scaling_law(self):
        assert GeneralScalingLaw is not None


class TestInstantiation:
    """Test that all scaling law classes can be instantiated."""

    def test_chinchilla_scaling_law_instantiation(self):
        law = ChinchillaScalingLaw()
        assert law is not None

    def test_basic_scaling_law_instantiation(self):
        law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        assert law is not None

    def test_data_constrained_scaling_law_instantiation(self):
        law = DataConstrainedScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        assert law is not None

    def test_general_scaling_law_instantiation(self):
        law = GeneralScalingLaw(
            params={'A': 1.0, 'B': 1.0, 'E': 1.0, 'alpha': 0.5, 'beta': 0.5},
            form_str='A / N**alpha + B / D**beta + E',
            params_str='A B E alpha beta',
            vars_str='N D',
            form_exp_parts_str=["logA - alpha * logN", "logB - beta * logD", "logE"],
        )
        assert law is not None


class TestTorchLossSignatures:
    """Test that torch_loss methods have consistent signatures."""

    @pytest.fixture
    def basic_torch_input(self):
        return {
            'N': torch.tensor([1e9, 2e9, 3e9], dtype=torch.float32),
            'D': torch.tensor([1e10, 2e10, 3e10], dtype=torch.float32),
            'Loss': torch.tensor([2.0, 1.8, 1.7], dtype=torch.float32),
        }

    @pytest.fixture
    def basic_params_list(self):
        return torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32, requires_grad=True)

    def test_chinchilla_torch_loss_signature(self, basic_torch_input, basic_params_list):
        """Test ChinchillaScalingLaw.torch_loss accepts the standard signature."""
        law = ChinchillaScalingLaw()
        loss = law.torch_loss(
            params_list=basic_params_list,
            form_exp_parts=law.apply_form_exp_parts,
            inp=basic_torch_input,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_basic_torch_loss_signature(self, basic_torch_input, basic_params_list):
        """Test BasicScalingLaw.torch_loss accepts the standard signature."""
        law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        loss = law.torch_loss(
            params_list=basic_params_list,
            form_exp_parts=law.apply_form_exp_parts,
            inp=basic_torch_input,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_data_constrained_torch_loss_signature(self):
        """Test DataConstrainedScalingLaw.torch_loss accepts the standard signature."""
        law = DataConstrainedScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        inp = {
            'UN': torch.tensor([1e9, 2e9, 3e9], dtype=torch.float32),
            'U': torch.tensor([1e10, 1e10, 1e10], dtype=torch.float32),
            'RD': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'RN': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
            'Loss': torch.tensor([2.0, 1.8, 1.7], dtype=torch.float32),
        }
        params_list = torch.tensor(
            [1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0], dtype=torch.float32, requires_grad=True
        )
        loss = law.torch_loss(
            params_list=params_list,
            form_exp_parts=law.form_exp_parts,
            inp=inp,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_general_torch_loss_signature(self, basic_torch_input, basic_params_list):
        """Test GeneralScalingLaw.torch_loss accepts the standard signature."""
        law = GeneralScalingLaw(
            params={'A': 1.0, 'B': 1.0, 'E': 1.0, 'alpha': 0.5, 'beta': 0.5},
            form_str='A / N**alpha + B / D**beta + E',
            params_str='A B E alpha beta',
            vars_str='N D',
            form_exp_parts_str=["logA - alpha * logN", "logB - beta * logD", "logE"],
        )

        # GeneralScalingLaw now uses apply_form_exp_parts with the same signature as ChinchillaScalingLaw
        loss = law.torch_loss(
            params_list=basic_params_list,
            form_exp_parts=law.apply_form_exp_parts,
            inp=basic_torch_input,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar


class TestNumpyLossSignatures:
    """Test that numpy_loss methods have consistent signatures."""

    @pytest.fixture
    def basic_numpy_input(self):
        return {
            'N': np.array([1e9, 2e9, 3e9], dtype=np.float32),
            'D': np.array([1e10, 2e10, 3e10], dtype=np.float32),
            'Loss': np.array([2.0, 1.8, 1.7], dtype=np.float32),
        }

    @pytest.fixture
    def basic_numpy_params(self):
        return np.array([1.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float32)

    def _basic_numpy_form_exp_parts(self, params, inp):
        """Helper function for numpy form_exp_parts."""
        logA, logB, logE, alpha, beta = params[0], params[1], params[2], params[3], params[4]
        N = inp['N']
        D = inp['D']
        return [
            logA - alpha * np.log(N),
            logB - beta * np.log(D),
            np.full(N.shape[0], logE),
        ]

    def test_chinchilla_numpy_loss_signature(self, basic_numpy_input, basic_numpy_params):
        """Test ChinchillaScalingLaw.numpy_loss accepts the standard signature."""
        law = ChinchillaScalingLaw()
        loss = law.numpy_loss(
            params_list=basic_numpy_params,
            form_exp_parts=lambda p, i: self._basic_numpy_form_exp_parts(p, i),
            inp=basic_numpy_input,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, (float, np.floating, np.ndarray))

    def test_basic_numpy_loss_signature(self, basic_numpy_input, basic_numpy_params):
        """Test BasicScalingLaw.numpy_loss accepts the standard signature."""
        law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        loss = law.numpy_loss(
            params_list=basic_numpy_params,
            form_exp_parts=lambda p, i: self._basic_numpy_form_exp_parts(p, i),
            inp=basic_numpy_input,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, (float, np.floating, np.ndarray))

    def test_data_constrained_numpy_loss_signature(self):
        """Test DataConstrainedScalingLaw.numpy_loss accepts the standard signature."""
        law = DataConstrainedScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        inp = {
            'UN': np.array([1e9, 2e9, 3e9], dtype=np.float32),
            'U': np.array([1e10, 1e10, 1e10], dtype=np.float32),
            'RD': np.array([0.1, 0.2, 0.3], dtype=np.float32),
            'RN': np.array([0.1, 0.2, 0.3], dtype=np.float32),
            'Loss': np.array([2.0, 1.8, 1.7], dtype=np.float32),
        }
        params_list = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0], dtype=np.float32)

        def dc_form_exp_parts(params, **inp_dict):
            a, b, e, alpha, beta, ep_star, n_star = params
            UN = inp_dict['UN']
            U = inp_dict['U']
            RD = inp_dict['RD']
            RN = inp_dict['RN']
            tm = UN + UN * n_star * (1 - np.exp(-RN / n_star))
            td = U + U * ep_star * (1 - np.exp(-RD / ep_star))
            return [
                a - alpha * np.log(tm),
                b - beta * np.log(td),
                np.full(inp_dict['Loss'].shape[0], e),
            ]

        loss = law.numpy_loss(
            params_list=params_list,
            form_exp_parts=dc_form_exp_parts,
            inp=inp,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, (float, np.floating, np.ndarray))

    def test_general_numpy_loss_signature(self, basic_numpy_input, basic_numpy_params):
        """Test GeneralScalingLaw.numpy_loss accepts the standard signature."""
        law = GeneralScalingLaw(
            params={'A': 1.0, 'B': 1.0, 'E': 1.0, 'alpha': 0.5, 'beta': 0.5},
            form_str='A / N**alpha + B / D**beta + E',
            params_str='A B E alpha beta',
            vars_str='N D',
            form_exp_parts_str=["logA - alpha * logN", "logB - beta * logD", "logE"],
        )
        loss = law.numpy_loss(
            params_list=basic_numpy_params,
            form_exp_parts=lambda p, inp: self._basic_numpy_form_exp_parts(p, inp),
            inp=basic_numpy_input,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, (float, np.floating, np.ndarray))


class TestLossFunctionOptions:
    """Test that different loss function options work."""

    @pytest.fixture
    def law_and_inputs(self):
        law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        inp = {
            'N': torch.tensor([1e9, 2e9, 3e9], dtype=torch.float32),
            'D': torch.tensor([1e10, 2e10, 3e10], dtype=torch.float32),
            'Loss': torch.tensor([2.0, 1.8, 1.7], dtype=torch.float32),
        }
        params = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32)
        return law, inp, params

    def test_log_huber_loss(self, law_and_inputs):
        law, inp, params = law_and_inputs
        loss = law.torch_loss(
            params_list=params,
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, torch.Tensor)

    def test_huber_loss(self, law_and_inputs):
        law, inp, params = law_and_inputs
        loss = law.torch_loss(
            params_list=params,
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
            loss_kwargs={'loss_func': 'huber', 'delta': 1e-3}
        )
        assert isinstance(loss, torch.Tensor)

    def test_log_mae_loss(self, law_and_inputs):
        law, inp, params = law_and_inputs
        loss = law.torch_loss(
            params_list=params,
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
            loss_kwargs={'loss_func': 'log_mae'}
        )
        assert isinstance(loss, torch.Tensor)

    def test_log_mse_loss(self, law_and_inputs):
        law, inp, params = law_and_inputs
        loss = law.torch_loss(
            params_list=params,
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
            loss_kwargs={'loss_func': 'log_mse'}
        )
        assert isinstance(loss, torch.Tensor)

    def test_unsupported_loss_raises(self, law_and_inputs):
        law, inp, params = law_and_inputs
        with pytest.raises(NotImplementedError):
            law.torch_loss(
                params_list=params,
                form_exp_parts=law.apply_form_exp_parts,
                inp=inp,
                loss_kwargs={'loss_func': 'unsupported_loss'}
            )


class TestTieIndices:
    """Test that tie_indices parameter works correctly."""

    def test_tie_indices_ties_parameters(self):
        """Test that tie_indices correctly ties parameter values."""
        law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        inp = {
            'N': torch.tensor([1e9, 2e9, 3e9], dtype=torch.float32),
            'D': torch.tensor([1e10, 2e10, 3e10], dtype=torch.float32),
            'Loss': torch.tensor([2.0, 1.8, 1.7], dtype=torch.float32),
        }
        # params: [logA, logB, logE, alpha, beta]
        # tie alpha (index 3) and beta (index 4)
        params = torch.tensor([1.0, 1.0, 0.5, 0.7, 0.3], dtype=torch.float32)

        # Without tie_indices, alpha=0.7, beta=0.3
        loss_no_tie = law.torch_loss(
            params_list=params.clone(),
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
            tie_indices=[],
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )

        # With tie_indices, alpha and beta should both be 0.7
        loss_with_tie = law.torch_loss(
            params_list=params.clone(),
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
            tie_indices=[[3, 4]],  # tie beta to alpha
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )

        # The losses should be different since the params are different
        assert loss_no_tie.item() != loss_with_tie.item()


class TestDefaultArguments:
    """Test that default arguments work correctly."""

    def test_torch_loss_default_tie_indices(self):
        """Test that torch_loss works with default empty tie_indices."""
        law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        inp = {
            'N': torch.tensor([1e9], dtype=torch.float32),
            'D': torch.tensor([1e10], dtype=torch.float32),
            'Loss': torch.tensor([2.0], dtype=torch.float32),
        }
        params = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32)

        # Call without tie_indices - should use default []
        loss = law.torch_loss(
            params_list=params,
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
            loss_kwargs={'loss_func': 'log_huber', 'delta': 1e-3}
        )
        assert isinstance(loss, torch.Tensor)

    def test_torch_loss_default_loss_kwargs(self):
        """Test that torch_loss works with default loss_kwargs."""
        law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'irreducible': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        inp = {
            'N': torch.tensor([1e9], dtype=torch.float32),
            'D': torch.tensor([1e10], dtype=torch.float32),
            'Loss': torch.tensor([2.0], dtype=torch.float32),
        }
        params = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5], dtype=torch.float32)

        # Call without loss_kwargs - should use default
        loss = law.torch_loss(
            params_list=params,
            form_exp_parts=law.apply_form_exp_parts,
            inp=inp,
        )
        assert isinstance(loss, torch.Tensor)


class TestGeneralScalingLawMatchesChinchilla:
    """
    Test that GeneralScalingLaw produces similar results to ChinchillaScalingLaw
    when initialized with the same form strings.

    The results won't be exactly identical because:
    - GeneralScalingLaw uses numerical root-finding for N_to_D and DL_to_N
    - ChinchillaScalingLaw uses analytic solutions
    """

    @pytest.fixture
    def sample_params(self):
        return {'A': 400.0, 'B': 2000.0, 'E': 1.7, 'alpha': 0.34, 'beta': 0.28}

    @pytest.fixture
    def chinchilla_law(self, sample_params):
        return ChinchillaScalingLaw(params=sample_params)

    @pytest.fixture
    def general_law_with_chinchilla_form(self, sample_params):
        """Create a GeneralScalingLaw using the same form strings as ChinchillaScalingLaw."""
        return GeneralScalingLaw(
            params=sample_params,
            form_str="E + A / N**alpha + B / D**beta",
            params_str="A B E alpha beta",
            vars_str="N D",
            form_exp_parts_str=["logA - alpha * logN", "logB - beta * logD", "logE"],
        )

    def test_loss_computation_matches(self, chinchilla_law, general_law_with_chinchilla_form):
        """Loss computation should be identical (both use the same lambdified form)."""
        test_cases = [
            (1e6, 1e8),
            (1e8, 1e10),
            (1e10, 1e12),
            (1e7, 1e11),
        ]

        for N, D in test_cases:
            chinchilla_loss = chinchilla_law.loss(N=N, D=D)
            general_loss = general_law_with_chinchilla_form.loss(N=N, D=D)

            assert abs(chinchilla_loss - general_loss) < 1e-10, \
                f"Loss mismatch at N={N}, D={D}: Chinchilla={chinchilla_loss}, General={general_loss}"

    def test_N_to_D_similar(self, chinchilla_law, general_law_with_chinchilla_form):
        """N_to_D should produce similar results (numeric vs analytic)."""
        # Target loss must be achievable: L > E + A/N^alpha
        # With E=1.7, A=400, alpha=0.34:
        #   N=1e8 -> min loss ≈ 2.43, so target=3.0 works
        #   N=1e9 -> min loss ≈ 2.03, so target=2.5 works
        #   N=1e10 -> min loss ≈ 1.85, so target=2.2 works
        test_cases = [
            (1e8, 3.0),
            (1e9, 2.5),
            (1e10, 2.2),
        ]

        for N, target_loss in test_cases:
            chinchilla_D = chinchilla_law.N_to_D(N, target_loss)
            general_D = general_law_with_chinchilla_form.N_to_D(N, target_loss)

            # Should match within 0.01% (numeric root-finding tolerance)
            relative_error = abs(chinchilla_D - general_D) / chinchilla_D
            assert relative_error < 1e-4, \
                f"N_to_D mismatch at N={N}, target_loss={target_loss}: " \
                f"Chinchilla={chinchilla_D:.6e}, General={general_D:.6e}, " \
                f"relative_error={relative_error:.2e}"

            # Verify both produce the correct target loss
            chinchilla_verified_loss = chinchilla_law.loss(N=N, D=chinchilla_D)
            general_verified_loss = general_law_with_chinchilla_form.loss(N=N, D=general_D)

            assert abs(chinchilla_verified_loss - target_loss) < 1e-6, \
                f"Chinchilla N_to_D verification failed: got loss {chinchilla_verified_loss}"
            assert abs(general_verified_loss - target_loss) < 1e-6, \
                f"General N_to_D verification failed: got loss {general_verified_loss}"

    def test_DL_to_N_similar(self, chinchilla_law, general_law_with_chinchilla_form):
        """DL_to_N should produce similar results (numeric vs analytic)."""
        # Target loss must be achievable: L > E + B/D^beta
        # With E=1.7, B=2000, beta=0.28:
        #   D=1e12 -> min loss ≈ 2.57, so target=3.0 works
        #   D=1e13 -> min loss ≈ 2.24, so target=2.5 works
        #   D=1e14 -> min loss ≈ 2.01, so target=2.2 works
        test_cases = [
            (1e12, 3.0),
            (1e13, 2.5),
            (1e14, 2.2),
        ]

        for D, target_loss in test_cases:
            chinchilla_N = chinchilla_law.DL_to_N(D, target_loss)
            general_N = general_law_with_chinchilla_form.DL_to_N(D, target_loss)

            # Should match within 0.01%
            relative_error = abs(chinchilla_N - general_N) / chinchilla_N
            assert relative_error < 1e-4, \
                f"DL_to_N mismatch at D={D}, target_loss={target_loss}: " \
                f"Chinchilla={chinchilla_N:.6e}, General={general_N:.6e}, " \
                f"relative_error={relative_error:.2e}"

            # Verify both produce the correct target loss
            chinchilla_verified_loss = chinchilla_law.loss(N=chinchilla_N, D=D)
            general_verified_loss = general_law_with_chinchilla_form.loss(N=general_N, D=D)

            assert abs(chinchilla_verified_loss - target_loss) < 1e-6, \
                f"Chinchilla DL_to_N verification failed: got loss {chinchilla_verified_loss}"
            assert abs(general_verified_loss - target_loss) < 1e-6, \
                f"General DL_to_N verification failed: got loss {general_verified_loss}"

    def test_optim_params_names_inferred_correctly(self, general_law_with_chinchilla_form):
        """optim_params_names should be correctly inferred from form_exp_parts_str."""
        expected = sorted(['alpha', 'beta', 'logA', 'logB', 'logE'])
        actual = sorted(general_law_with_chinchilla_form.optim_params_names)

        assert actual == expected, \
            f"optim_params_names mismatch: expected {expected}, got {actual}"

    def test_param_names_match(self, chinchilla_law, general_law_with_chinchilla_form):
        """param_names should match between both implementations."""
        assert chinchilla_law.param_names == general_law_with_chinchilla_form.param_names

    def test_var_names_match(self, chinchilla_law, general_law_with_chinchilla_form):
        """var_names should match between both implementations."""
        assert chinchilla_law.var_names == general_law_with_chinchilla_form.var_names

    def test_compute_optimal_allocation_similar(self, chinchilla_law, general_law_with_chinchilla_form):
        """compute_optimal_allocation should produce similar results."""
        compute_budgets = [1e18, 1e20, 1e22]

        for C in compute_budgets:
            chinchilla_result = chinchilla_law.compute_optimal_allocation(C=C)
            general_result = general_law_with_chinchilla_form.compute_optimal_allocation(C=C)

            # Model size should match closely
            N_relative_error = abs(chinchilla_result['model'] - general_result['model']) / chinchilla_result['model']
            assert N_relative_error < 1e-3, \
                f"Optimal N mismatch at C={C}: " \
                f"Chinchilla={chinchilla_result['model']:.2e}, General={general_result['model']:.2e}"

            # Data size should match closely
            D_relative_error = abs(chinchilla_result['data'] - general_result['data']) / chinchilla_result['data']
            assert D_relative_error < 1e-3, \
                f"Optimal D mismatch at C={C}: " \
                f"Chinchilla={chinchilla_result['data']:.2e}, General={general_result['data']:.2e}"

            # Loss should match closely
            loss_relative_error = abs(chinchilla_result['loss'] - general_result['loss']) / chinchilla_result['loss']
            assert loss_relative_error < 1e-6, \
                f"Optimal loss mismatch at C={C}: " \
                f"Chinchilla={chinchilla_result['loss']:.6f}, General={general_result['loss']:.6f}"
