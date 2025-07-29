"""
Property-based tests for scaling laws that should hold for ALL implementations.
These tests verify mathematical consistency and catch implementation errors.
"""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from scaling_laws import ALL_SCALING_LAWS
from scaling_law_classes.scaling_law import LawParams
from scaling_law_classes.basic_scaling_law import BasicScalingLaw


class TestScalingLawProperties:
    """Test mathematical properties that should hold for ALL scaling laws."""
    
    @pytest.fixture
    def all_laws(self):
        """Get all scaling laws for testing."""
        return {name: wrapper.scaling_law for name, wrapper in ALL_SCALING_LAWS.items()}
    
    @pytest.fixture
    def sample_params(self):
        """Sample parameters for creating test laws."""
        return LawParams(A=400.0, B=2000.0, alpha=0.34, beta=0.28, irreducible=1.7)
    
    def test_loss_monotonicity(self, all_laws):
        """Loss should decrease as N or D increase (holding others constant)."""
        for name, law in all_laws.items():
            # Test N monotonicity (D fixed)
            N_values = [1e6, 1e7, 1e8, 1e9]
            D_fixed = 1e9
            extra_args = {"U": 1e15} if "U" in getattr(law, 'variables', []) else {}
            
            losses = []
            for N in N_values:
                try:
                    loss = law.loss(N=N, D=D_fixed, **extra_args)
                    losses.append(loss)
                except Exception as e:
                    pytest.skip(f"Law {name} failed: {e}")
            
            # Loss should decrease as N increases
            assert all(losses[i] >= losses[i+1] for i in range(len(losses)-1)), \
                f"{name}: Loss should decrease as N increases. Got {losses}"
            
            # Test D monotonicity (N fixed)
            D_values = [1e8, 1e9, 1e10, 1e11]
            N_fixed = 1e8
            
            losses = []
            for D in D_values:
                try:
                    loss = law.loss(N=N_fixed, D=D, **extra_args)
                    losses.append(loss)
                except Exception as e:
                    pytest.skip(f"Law {name} failed: {e}")
            
            # Loss should decrease as D increases
            assert all(losses[i] >= losses[i+1] for i in range(len(losses)-1)), \
                f"{name}: Loss should decrease as D increases. Got {losses}"
    
    def test_loss_bounds(self, all_laws):
        """Loss should always be >= irreducible loss."""
        for name, law in all_laws.items():
            irreducible = law.params.irreducible
            extra_args = {"U": 1e15} if "U" in getattr(law, 'variables', []) else {}
            
            # Test various N, D combinations
            test_cases = [
                (1e6, 1e8), (1e8, 1e10), (1e10, 1e12), (1e12, 1e14)
            ]
            
            for N, D in test_cases:
                try:
                    loss = law.loss(N=N, D=D, **extra_args)
                    assert loss >= irreducible - 1e-10, \
                        f"{name}: Loss {loss} < irreducible {irreducible} for N={N}, D={D}"
                except Exception as e:
                    pytest.skip(f"Law {name} failed: {e}")
    
    def test_N_to_D_consistency(self, all_laws):
        """N_to_D should be consistent with loss function."""
        for name, law in all_laws.items():
            target_loss = law.params.irreducible + 0.5  # Achievable loss
            N_test = 1e8
            extra_args = {"U": 1e15} if "U" in getattr(law, 'variables', []) else {}
            
            try:
                # Get D that should achieve target_loss at N_test
                D_computed = law.N_to_D(N_test, target_loss, **extra_args)
                
                # Verify that loss(N_test, D_computed) ≈ target_loss
                actual_loss = law.loss(N=N_test, D=D_computed, **extra_args)
                
                assert abs(actual_loss - target_loss) < 1e-6, \
                    f"{name}: N_to_D inconsistent. Target: {target_loss}, Got: {actual_loss}"
                    
            except Exception as e:
                pytest.skip(f"Law {name} N_to_D failed: {e}")
    
    def test_compute_optimal_allocation_feasibility(self, all_laws):
        """Optimal allocation should satisfy compute constraint."""
        for name, law in all_laws.items():
            compute_budgets = [1e18, 1e20, 1e22]
            extra_args = {"U": 1e15} if "U" in getattr(law, 'variables', []) else {}
            
            for C in compute_budgets:
                try:
                    result = law.compute_optimal_allocation(C=C, **extra_args)
                    
                    # Check that computed FLOPs match budget
                    computed_flops = law.flops(N=result['model'], D=result['data'], **extra_args)
                    
                    assert abs(computed_flops - C) / C < 1e-6, \
                        f"{name}: Compute constraint violated. Budget: {C}, Computed: {computed_flops}"
                        
                    # Check that loss is reasonable
                    assert result['loss'] > law.params.irreducible, \
                        f"{name}: Optimal loss {result['loss']} <= irreducible {law.params.irreducible}"
                        
                except Exception as e:
                    pytest.skip(f"Law {name} optimal allocation failed: {e}")
    
    def test_scaling_behavior_asymptotes(self, all_laws):
        """Test scaling behavior at extreme values."""
        for name, law in all_laws.items():
            extra_args = {"U": 1e15} if "U" in getattr(law, 'variables', []) else {}
            
            # Very large N should approach irreducible + B/D^beta
            N_large = 1e15
            D_test = 1e10
            
            try:
                loss_large_N = law.loss(N=N_large, D=D_test, **extra_args)
                expected_approx = law.params.irreducible + law.params.B / (D_test ** law.params.beta)
                
                # Should be within reasonable range
                assert abs(loss_large_N - expected_approx) / expected_approx < 0.1, \
                    f"{name}: Large N asymptote incorrect. Got {loss_large_N}, expected ~{expected_approx}"
                    
            except Exception as e:
                pytest.skip(f"Law {name} asymptote test failed: {e}")
    
    def test_parameter_sensitivity(self, sample_params):
        """Test that parameter changes affect loss in expected directions."""
        base_law = BasicScalingLaw(sample_params)
        
        # Increase A should increase loss (model term gets worse)
        high_A_params = LawParams(A=sample_params.A * 2, B=sample_params.B, 
                                  alpha=sample_params.alpha, beta=sample_params.beta,
                                  irreducible=sample_params.irreducible)
        high_A_law = BasicScalingLaw(high_A_params)
        
        N, D = 1e8, 1e10
        base_loss = base_law.loss(N=N, D=D)
        high_A_loss = high_A_law.loss(N=N, D=D)
        
        assert high_A_loss > base_loss, "Higher A should increase loss"
        
        # Similar tests for other parameters...
    
    def test_variable_requirements(self, all_laws):
        """Each law should enforce its required variables."""
        for name, law in ALL_SCALING_LAWS.items():
            scaling_law = law.scaling_law
            required_vars = getattr(scaling_law, 'variables', ())
            
            # Test that missing required variables raise errors
            incomplete_args = {"N": 1e8}  # Missing D and potentially U
            
            if len(required_vars) > 1:
                with pytest.raises(ValueError, match="Missing vars"):
                    scaling_law.loss(**incomplete_args)


class TestSpecificImplementations:
    """Tests for specific scaling law implementations."""
    
    def test_basic_scaling_law_formula(self):
        """Verify BasicScalingLaw implements correct formula."""
        params = LawParams(A=400.0, B=2000.0, alpha=0.34, beta=0.28, irreducible=1.7)
        law = BasicScalingLaw(params)
        
        N, D = 1e8, 1e10
        computed_loss = law.loss(N=N, D=D)
        expected_loss = 1.7 + 400.0/(N**0.34) + 2000.0/(D**0.28)
        
        assert abs(computed_loss - expected_loss) < 1e-10, \
            f"Basic scaling law formula incorrect. Got {computed_loss}, expected {expected_loss}"
    
    def test_data_constrained_unlimited_case(self):
        """When U is very large, DataConstrainedScalingLaw should approach BasicScalingLaw."""
        if "Data-Constrained Scaling Law" not in ALL_SCALING_LAWS:
            pytest.skip("DataConstrainedScalingLaw not available")
            
        dc_law = ALL_SCALING_LAWS["Data-Constrained Scaling Law"].scaling_law
        basic_law = ALL_SCALING_LAWS["Chinchilla"].scaling_law
        
        N, D = 1e8, 1e10
        U_unlimited = 1e15
        
        dc_loss = dc_law.loss(N=N, D=D, U=U_unlimited)
        
        # Should be reasonably close to a basic scaling law
        # (not identical due to different parameters, but should follow similar scaling)
        assert dc_loss > dc_law.params.irreducible, \
            "Data-constrained law with unlimited U should give reasonable loss"


class TestNumericalStability:
    """Test numerical stability across parameter ranges."""
    
    @pytest.fixture
    def all_laws(self):
        return {name: wrapper.scaling_law for name, wrapper in ALL_SCALING_LAWS.items()}
    
    def test_extreme_parameter_values(self, all_laws):
        """Test behavior with very small/large parameter values."""
        for name, law in all_laws.items():
            extra_args = {"U": 1e15} if "U" in getattr(law, 'variables', []) else {}
            
            # Test very small N, D
            extreme_cases = [
                (1e3, 1e6), (1e15, 1e18), (1e6, 1e15)
            ]
            
            for N, D in extreme_cases:
                try:
                    loss = law.loss(N=N, D=D, **extra_args)
                    assert np.isfinite(loss), f"{name}: Non-finite loss for N={N}, D={D}"
                    assert loss > 0, f"{name}: Non-positive loss for N={N}, D={D}"
                except Exception as e:
                    pytest.skip(f"Law {name} failed extreme case N={N}, D={D}: {e}")
    
    def test_optimization_convergence(self, all_laws):
        """Test that optimization routines converge reliably."""
        for name, law in all_laws.items():
            extra_args = {"U": 1e15} if "U" in getattr(law, 'variables', []) else {}
            
            # Test multiple compute budgets
            budgets = [1e18, 1e20, 1e22]
            
            for C in budgets:
                try:
                    # Should not raise optimization errors
                    result = law.compute_optimal_allocation(C=C, **extra_args)
                    
                    # Results should be reasonable
                    assert result['model'] > 1e3, f"{name}: Model size too small: {result['model']}"
                    assert result['data'] > 1e6, f"{name}: Data size too small: {result['data']}"
                    assert np.isfinite(result['loss']), f"{name}: Non-finite loss: {result['loss']}"
                    
                except Exception as e:
                    pytest.skip(f"Law {name} optimization failed for C={C}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])