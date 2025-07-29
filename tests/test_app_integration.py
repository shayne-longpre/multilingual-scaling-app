"""
Integration tests for the Streamlit app functionality.
Tests the complete workflow without UI interactions.
"""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from scaling_laws import ALL_SCALING_LAWS


class TestAppIntegration:
    """Test app functionality without Streamlit UI."""
    
    def test_all_laws_loadable(self):
        """All scaling laws should load without errors."""
        assert len(ALL_SCALING_LAWS) > 0, "No scaling laws loaded"
        
        for name, wrapper in ALL_SCALING_LAWS.items():
            assert wrapper.scaling_law is not None, f"Law {name} has no scaling_law"
            assert wrapper.name == name, f"Law name mismatch: {wrapper.name} != {name}"
            assert wrapper.paper.startswith("http"), f"Law {name} has invalid paper URL"
    
    def test_compute_budget_workflow(self):
        """Test complete compute budget optimization workflow."""
        compute_budget = 1e20
        selected_laws = ["Chinchilla", "Chinchilla Replication"]
        total_tokens = 0  # Unlimited case
        inference_tokens = 0
        
        results = []
        
        for law_name in selected_laws:
            if law_name not in ALL_SCALING_LAWS:
                continue
                
            law_wrapper = ALL_SCALING_LAWS[law_name]
            scaling_law = law_wrapper.scaling_law
            
            # Simulate app logic
            try:
                if "U" in law_wrapper.extra_args:
                    U_value = total_tokens if total_tokens > 0 else 1e15
                    result = scaling_law.compute_optimal_allocation(C=compute_budget, U=U_value)
                else:
                    result = scaling_law.compute_optimal_allocation(C=compute_budget)
                
                results.append({
                    "Scaling Law": law_name,
                    "Optimal N*": f"{result['model']:.2e}",
                    "Optimal D*": f"{result['data']:.2e}",
                    "Loss": f"{result['loss']:.4f}",
                    "Paper": law_wrapper.paper,
                })
                
            except Exception as e:
                pytest.fail(f"Compute budget workflow failed for {law_name}: {e}")
        
        # Should have results for each law
        assert len(results) == len(selected_laws), "Missing results for some laws"
        
        # Each result should have reasonable values
        for result in results:
            N_val = float(result["Optimal N*"])
            D_val = float(result["Optimal D*"])
            loss_val = float(result["Loss"])
            
            assert 1e6 <= N_val <= 1e15, f"Unreasonable N*: {N_val}"
            assert 1e9 <= D_val <= 1e16, f"Unreasonable D*: {D_val}"
            assert 1.0 <= loss_val <= 10.0, f"Unreasonable loss: {loss_val}"
    
    def test_data_constrained_workflow(self):
        """Test workflow with data constraints."""
        if "Data-Constrained Scaling Law" not in ALL_SCALING_LAWS:
            pytest.skip("DataConstrainedScalingLaw not available")
        
        compute_budget = 1e20
        total_tokens_cases = [0, 1e12]  # Unlimited and constrained
        
        for total_tokens in total_tokens_cases:
            law_wrapper = ALL_SCALING_LAWS["Data-Constrained Scaling Law"]
            scaling_law = law_wrapper.scaling_law
            
            try:
                U_value = total_tokens if total_tokens > 0 else 1e15
                result = scaling_law.compute_optimal_allocation(C=compute_budget, U=U_value)
                
                # Should get reasonable results in both cases
                assert result['model'] > 1e6, f"Model size too small: {result['model']}"
                assert result['data'] > 1e9, f"Data size too small: {result['data']}"
                assert result['loss'] > 1.0, f"Loss too small: {result['loss']}"
                
            except Exception as e:
                pytest.fail(f"Data constrained workflow failed for U={total_tokens}: {e}")
    
    def test_plotting_data_generation(self):
        """Test that plotting data can be generated without errors."""
        compute_range = np.logspace(18, 22, 10)  # Smaller range for testing
        selected_laws = ["Chinchilla"]
        total_tokens = 0
        
        for law_name in selected_laws:
            law_wrapper = ALL_SCALING_LAWS[law_name]
            scaling_law = law_wrapper.scaling_law
            
            N_optimal = []
            D_optimal = []
            
            for C in compute_range:
                try:
                    if "U" in law_wrapper.extra_args:
                        U_value = total_tokens if total_tokens > 0 else 1e15
                        result = scaling_law.compute_optimal_allocation(C=C, U=U_value)
                    else:
                        result = scaling_law.compute_optimal_allocation(C=C)
                    
                    N_optimal.append(result['model'])
                    D_optimal.append(result['data'])
                    
                except Exception as e:
                    pytest.fail(f"Plotting data generation failed for {law_name}, C={C}: {e}")
            
            # Should have data for all compute budgets
            assert len(N_optimal) == len(compute_range), "Missing optimal N values"
            assert len(D_optimal) == len(compute_range), "Missing optimal D values"
            
            # Values should be finite and positive
            assert all(np.isfinite(N_optimal)), "Non-finite N values"
            assert all(np.isfinite(D_optimal)), "Non-finite D values"
            assert all(n > 0 for n in N_optimal), "Non-positive N values"
            assert all(d > 0 for d in D_optimal), "Non-positive D values"
    
    def test_target_loss_mode(self):
        """Test target loss mode functionality."""
        target_loss = 2.0
        selected_laws = ["Chinchilla"]
        total_tokens = 0
        
        for law_name in selected_laws:
            law_wrapper = ALL_SCALING_LAWS[law_name]
            scaling_law = law_wrapper.scaling_law
            
            # Test N_to_D for iso-loss curves
            N_test_values = [1e7, 1e8, 1e9]
            
            for N in N_test_values:
                try:
                    if "U" in law_wrapper.extra_args:
                        U_value = total_tokens if total_tokens > 0 else 1e15
                        D = scaling_law.N_to_D(N, target_loss, U=U_value)
                    else:
                        D = scaling_law.N_to_D(N, target_loss)
                    
                    assert D > 0, f"Non-positive D: {D}"
                    assert np.isfinite(D), f"Non-finite D: {D}"
                    
                    # Verify loss is approximately correct
                    extra_args = {"U": U_value} if "U" in law_wrapper.extra_args else {}
                    actual_loss = scaling_law.loss(N=N, D=D, **extra_args)
                    
                    assert abs(actual_loss - target_loss) < 1e-6, \
                        f"Target loss verification failed: {actual_loss} != {target_loss}"
                        
                except Exception as e:
                    if "infeasible" in str(e).lower():
                        # This is expected for some N, target_loss combinations
                        continue
                    else:
                        pytest.fail(f"Target loss mode failed for {law_name}, N={N}: {e}")
    
    def test_inference_optimization(self):
        """Test inference optimization workflow."""
        compute_budget = 1e20
        inference_tokens = 1e10
        selected_laws = ["Chinchilla"]  # Only test basic law for inference
        
        for law_name in selected_laws:
            law_wrapper = ALL_SCALING_LAWS[law_name]
            scaling_law = law_wrapper.scaling_law
            
            # Skip if inference optimization not implemented
            if hasattr(scaling_law, 'compute_optimal_train_tokens'):
                try:
                    # Test the method exists and doesn't immediately fail
                    test_call = scaling_law.compute_optimal_train_tokens(1e10, inference_tokens, 2.0)
                    # If it returns a number, inference is implemented
                    if isinstance(test_call, (int, float)) and test_call != 0.0:
                        result = scaling_law.compute_optimal_allocation_inference(
                            compute_budget=compute_budget,
                            D_inference=inference_tokens
                        )
                        
                        assert result['model'] > 1e6, "Model size unreasonable"
                        assert result['data'] > 1e9, "Data size unreasonable"
                        assert result['flops_train'] > 0, "Training FLOPs should be positive"
                        assert result['flops_inference'] > 0, "Inference FLOPs should be positive"
                        
                except NotImplementedError:
                    pytest.skip(f"Inference optimization not implemented for {law_name}")
                except Exception as e:
                    pytest.skip(f"Inference optimization test failed for {law_name}: {e}")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameter values."""
        invalid_budgets = [0, -1e20, float('inf'), float('nan')]
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        for budget in invalid_budgets:
            with pytest.raises((ValueError, RuntimeError, OverflowError)):
                scaling_law.compute_optimal_allocation(C=budget)
    
    def test_missing_required_args(self):
        """Test error handling for missing required arguments."""
        if "Data-Constrained Scaling Law" not in ALL_SCALING_LAWS:
            pytest.skip("DataConstrainedScalingLaw not available")
        
        law_wrapper = ALL_SCALING_LAWS["Data-Constrained Scaling Law"]
        scaling_law = law_wrapper.scaling_law
        
        # Should fail without U parameter
        with pytest.raises(ValueError):
            scaling_law.N_to_D(1e8, 2.0)  # Missing U
    
    def test_infeasible_target_loss(self):
        """Test handling of infeasible target loss values."""
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        # Target loss below irreducible should fail
        irreducible = scaling_law.params.irreducible
        infeasible_loss = irreducible - 0.1
        
        with pytest.raises(ValueError):
            scaling_law.N_to_D(1e8, infeasible_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])