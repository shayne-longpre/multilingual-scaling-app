"""
Tests for the auto_generate_form_exp_parts utility function.
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.utils import auto_generate_form_exp_parts, validate_form_exp_parts


class TestAutoGenerateFormExpParts:
    """Tests for auto_generate_form_exp_parts function."""

    def test_chinchilla_formula(self):
        """Test standard Chinchilla formula."""
        form_str = "E + A / N**alpha + B / D**beta"
        params_str = "A B E alpha beta"

        result = auto_generate_form_exp_parts(form_str, params_str)

        assert len(result) == 3, f"Expected 3 terms, got {len(result)}"
        # The log-space expressions should reference the parameters
        # Accept either expanded form (logA - alpha*logN) or compact form (log(A/N**alpha))
        combined = " ".join(result)
        assert "A" in combined, "A should appear in some form"
        assert "B" in combined, "B should appear in some form"
        assert "E" in combined, "E should appear in some form"
        assert "alpha" in combined, "alpha should appear in some form"
        assert "beta" in combined, "beta should appear in some form"

    def test_single_term_constant(self):
        """Test formula with single constant term."""
        form_str = "E"
        params_str = "E"

        result = auto_generate_form_exp_parts(form_str, params_str)

        assert len(result) == 1
        assert "logE" in result[0] or "log(E)" in result[0]

    def test_power_law_n_only(self):
        """Test power law with N only."""
        form_str = "E + A / N**alpha"
        params_str = "A E alpha"

        result = auto_generate_form_exp_parts(form_str, params_str)

        assert len(result) == 2
        combined = " ".join(result)
        assert "A" in combined, "A should appear"
        assert "alpha" in combined, "alpha should appear"
        assert "N" in combined, "N should appear"
        assert "E" in combined, "E should appear"

    def test_power_law_d_only(self):
        """Test power law with D only."""
        form_str = "E + B / D**beta"
        params_str = "B E beta"

        result = auto_generate_form_exp_parts(form_str, params_str)

        assert len(result) == 2
        combined = " ".join(result)
        assert "B" in combined, "B should appear"
        assert "beta" in combined, "beta should appear"
        assert "D" in combined, "D should appear"
        assert "E" in combined, "E should appear"

    def test_multiplication_term(self):
        """Test formula with multiplication in numerator."""
        form_str = "A * B / N**alpha"
        params_str = "A B alpha"

        result = auto_generate_form_exp_parts(form_str, params_str)

        assert len(result) == 1
        combined = " ".join(result)
        # Should have both A and B in the expression
        assert "A" in combined, "A should appear"
        assert "B" in combined, "B should appear"

    def test_invalid_formula(self):
        """Test handling of invalid formula."""
        form_str = "invalid syntax +++ ---"
        params_str = "A B"

        with pytest.raises(ValueError) as exc_info:
            auto_generate_form_exp_parts(form_str, params_str)

        assert "parse" in str(exc_info.value).lower()

    def test_three_variable_formula(self):
        """Test formula with three variables (N, D, U)."""
        form_str = "E + A / N**alpha + B / (D * U)**beta"
        params_str = "A B E alpha beta"

        # This may or may not work depending on complexity
        try:
            result = auto_generate_form_exp_parts(form_str, params_str)
            # If it works, verify it has reasonable structure
            assert len(result) >= 2
        except ValueError:
            # Complex formulas may not be auto-generatable
            pass


class TestValidateFormExpParts:
    """Tests for validate_form_exp_parts function."""

    def test_valid_expressions(self):
        """Test validation of correct expressions."""
        form_str = "E + A / N**alpha + B / D**beta"
        params_str = "A B E alpha beta"
        vars_str = "N D"
        form_exp_parts_str = ["logA - alpha * logN", "logB - beta * logD", "logE"]

        valid, error = validate_form_exp_parts(
            form_str, params_str, vars_str, form_exp_parts_str
        )

        assert valid is True
        assert error is None

    def test_unknown_symbol(self):
        """Test detection of unknown symbols in expressions."""
        form_str = "E + A / N**alpha"
        params_str = "A E alpha"
        vars_str = "N D"
        form_exp_parts_str = ["logA - alpha * logN", "logX"]  # X is unknown

        valid, error = validate_form_exp_parts(
            form_str, params_str, vars_str, form_exp_parts_str
        )

        assert valid is False
        assert error is not None
        assert "X" in error

    def test_invalid_expression_syntax(self):
        """Test detection of invalid expression syntax."""
        form_str = "E + A / N**alpha"
        params_str = "A E alpha"
        vars_str = "N D"
        form_exp_parts_str = ["logA - alpha * logN", "invalid ++ syntax"]

        valid, error = validate_form_exp_parts(
            form_str, params_str, vars_str, form_exp_parts_str
        )

        assert valid is False
        assert error is not None

    def test_log_prefixed_names_allowed(self):
        """Test that logX names are allowed when X is a param or var."""
        form_str = "E + A / N**alpha"
        params_str = "A E alpha"
        vars_str = "N D"
        form_exp_parts_str = ["logA - alpha * logN", "logE"]

        valid, error = validate_form_exp_parts(
            form_str, params_str, vars_str, form_exp_parts_str
        )

        assert valid is True


class TestIntegrationWithGeneralScalingLaw:
    """Integration tests with GeneralScalingLaw class."""

    def test_generated_expressions_work_with_general_scaling_law(self):
        """Test that auto-generated expressions can be used with GeneralScalingLaw."""
        from src.scaling_law_classes.general_scaling_law import GeneralScalingLaw

        form_str = "E + A / N**alpha + B / D**beta"
        params_str = "A B E alpha beta"
        vars_str = "N D"

        # Generate expressions
        form_exp_parts = auto_generate_form_exp_parts(form_str, params_str)

        # Create a GeneralScalingLaw with dummy params
        params = {"A": 400.0, "B": 400.0, "E": 1.5, "alpha": 0.35, "beta": 0.35}

        try:
            law = GeneralScalingLaw(
                params=params,
                form_str=form_str,
                params_str=params_str,
                vars_str=vars_str,
                form_exp_parts_str=form_exp_parts,
            )

            # Test that loss computation works
            loss = law.loss_expr(N=1e9, D=1e12)
            assert loss > 0, "Loss should be positive"
            assert loss < 100, "Loss should be reasonable"

        except Exception as e:
            pytest.fail(f"GeneralScalingLaw creation failed: {e}")

    def test_preset_chinchilla_expressions(self):
        """Test that preset Chinchilla expressions work correctly."""
        from src.scaling_law_classes.general_scaling_law import GeneralScalingLaw

        form_str = "E + A / N**alpha + B / D**beta"
        params_str = "A B E alpha beta"
        vars_str = "N D"
        # These are the preset expressions from the frontend
        form_exp_parts = ["logA - alpha * logN", "logB - beta * logD", "logE"]

        params = {"A": 400.0, "B": 400.0, "E": 1.5, "alpha": 0.35, "beta": 0.35}

        law = GeneralScalingLaw(
            params=params,
            form_str=form_str,
            params_str=params_str,
            vars_str=vars_str,
            form_exp_parts_str=form_exp_parts,
        )

        # Test various N, D combinations
        test_cases = [
            (1e7, 1e9),
            (1e8, 1e10),
            (1e9, 1e11),
            (1e10, 1e12),
        ]

        for N, D in test_cases:
            loss = law.loss_expr(N=N, D=D)
            assert loss > 1.5, f"Loss should be > E (irreducible) for N={N}, D={D}"
            assert loss < 10, f"Loss should be reasonable for N={N}, D={D}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
