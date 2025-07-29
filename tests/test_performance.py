"""
Performance and benchmark tests for scaling laws.
These tests ensure implementations are reasonably efficient.
"""
import pytest
import time
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from scaling_laws import ALL_SCALING_LAWS


class TestPerformance:
    """Performance benchmarks for scaling law operations."""
    
    def test_loss_computation_speed(self):
        """Loss computation should be fast for vectorized operations."""
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        # Test single computation
        start_time = time.time()
        for _ in range(1000):
            loss = scaling_law.loss(N=1e8, D=1e10)
        single_time = time.time() - start_time
        
        # Should be fast
        assert single_time < 1.0, f"Loss computation too slow: {single_time:.3f}s for 1000 calls"
    
    def test_optimization_speed(self):
        """Optimization should converge reasonably quickly."""
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        compute_budgets = [1e18, 1e20, 1e22]
        
        start_time = time.time()
        for C in compute_budgets:
            result = scaling_law.compute_optimal_allocation(C=C)
            assert result is not None
        optimization_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert optimization_time < 5.0, f"Optimization too slow: {optimization_time:.3f}s"
    
    def test_plotting_data_generation_speed(self):
        """Generating plotting data should be efficient."""
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        compute_range = np.logspace(15, 25, 100)
        
        start_time = time.time()
        for C in compute_range:
            result = scaling_law.compute_optimal_allocation(C=C)
        plotting_time = time.time() - start_time
        
        # Should be reasonable for 100 points
        assert plotting_time < 10.0, f"Plotting data generation too slow: {plotting_time:.3f}s"
    
    def test_memory_usage(self):
        """Operations should not leak memory significantly."""
        import tracemalloc
        
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        tracemalloc.start()
        
        # Perform many operations
        for _ in range(100):
            scaling_law.loss(N=1e8, D=1e10)
            scaling_law.compute_optimal_allocation(C=1e20)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Peak memory should be reasonable (less than 100MB)
        assert peak < 100 * 1024 * 1024, f"Memory usage too high: {peak / 1024 / 1024:.1f}MB"


class TestScalability:
    """Test scalability with different parameter ranges."""
    
    def test_extreme_parameter_ranges(self):
        """Should handle extreme but realistic parameter ranges."""
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        # Test extreme but realistic ranges
        extreme_cases = [
            (1e6, 1e9),    # Small model, small data
            (1e12, 1e15),  # Large model, large data
            (1e6, 1e15),   # Small model, large data
            (1e12, 1e9),   # Large model, small data
        ]
        
        for N, D in extreme_cases:
            try:
                loss = scaling_law.loss(N=N, D=D)
                assert np.isfinite(loss), f"Non-finite loss for extreme case N={N}, D={D}"
                assert loss > 0, f"Non-positive loss for extreme case N={N}, D={D}"
            except Exception as e:
                pytest.fail(f"Failed on extreme case N={N}, D={D}: {e}")
    
    def test_optimization_convergence_reliability(self):
        """Optimization should converge reliably across parameter ranges."""
        law_name = "Chinchilla"
        law_wrapper = ALL_SCALING_LAWS[law_name]
        scaling_law = law_wrapper.scaling_law
        
        # Test wide range of compute budgets
        budgets = np.logspace(16, 24, 20)
        
        failures = 0
        for C in budgets:
            try:
                result = scaling_law.compute_optimal_allocation(C=C)
                # Basic sanity checks
                assert result['model'] > 1e3
                assert result['data'] > 1e6
                assert np.isfinite(result['loss'])
            except Exception as e:
                failures += 1
                if failures > 2:  # Allow a few failures
                    pytest.fail(f"Too many optimization failures: {failures}/20")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])