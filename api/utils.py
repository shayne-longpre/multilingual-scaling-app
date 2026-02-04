import re

from sympy import Add, Mul, Pow, Symbol, symbols, log, simplify, parse_expr


def auto_generate_form_exp_parts(form_str: str, params_str: str) -> list[str]:
    """
    Auto-generate log-space expressions from a formula string.

    Given a formula like "E + A / N**alpha + B / D**beta", this function:
    1. Parses the formula into a SymPy expression
    2. Identifies additive terms
    3. Converts each term to a log-space expression

    Supported patterns:
    - A / N**alpha  → "logA - alpha * logN"
    - A * N**alpha  → "logA + alpha * logN"
    - E (constant)  → "logE"
    - A * B / (N**alpha * D**beta) → "logA + logB - alpha * logN - beta * logD"

    Args:
        form_str: The loss formula as a string (e.g., "E + A / N**alpha + B / D**beta")
        params_str: Space-separated parameter names (e.g., "A B E alpha beta")

    Returns:
        List of log-space expressions, one per additive term

    Raises:
        ValueError: If the formula cannot be parsed or contains unsupported patterns
    """
    param_names = set(params_str.split())

    try:
        # Extract all identifiers from the formula
        identifiers = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", form_str))

        # Create a local dict with all identifiers as symbols
        local_dict = {name: Symbol(name) for name in identifiers}

        # Parse with explicit local dict to avoid sympy interpreting N, D, E, etc as functions
        expr = parse_expr(form_str.strip(), local_dict=local_dict)
    except Exception as e:
        raise ValueError(f"Failed to parse formula: {e}") from e

    # Get additive terms
    if isinstance(expr, Add):
        terms = list(expr.args)
    else:
        terms = [expr]

    log_parts = []
    for term in terms:
        try:
            log_part = _term_to_log_space(term, param_names)
            log_parts.append(log_part)
        except ValueError as e:
            raise ValueError(f"Cannot convert term '{term}' to log-space: {e}") from e

    return log_parts


def _term_to_log_space(term, param_names: set[str]) -> str:
    """
    Convert a single term to its log-space representation.

    Uses SymPy's log simplification to handle the conversion.
    """
    # Get all symbols in the term
    term_symbols = term.free_symbols

    # Check if term is just a symbol (e.g., E)
    if isinstance(term, Symbol):
        return f"log{term.name}"

    # Take log and simplify
    log_term = log(term)

    try:
        simplified = simplify(log_term, force=True)
    except Exception:
        simplified = log_term.expand(log=True, force=True)

    # Convert to string and replace log(X) with logX
    result = str(simplified)

    # Replace log(X) patterns with logX for all symbols
    for sym in term_symbols:
        sym_name = sym.name
        result = result.replace(f"log({sym_name})", f"log{sym_name}")

    # Clean up: replace "1*" patterns
    result = result.replace("1*", "")

    return result


def validate_form_exp_parts(
    form_str: str, params_str: str, vars_str: str, form_exp_parts_str: list[str]
) -> tuple[bool, str | None]:
    """
    Validate that form_exp_parts_str expressions are valid for the given formula.

    Args:
        form_str: The loss formula
        params_str: Space-separated parameter names
        vars_str: Space-separated variable names
        form_exp_parts_str: List of log-space expressions to validate

    Returns:
        Tuple of (is_valid, error_message_or_none)
    """
    param_names = set(params_str.split())
    var_names = set(vars_str.split())
    all_names = param_names | var_names
    log_names = {f"log{n}" for n in all_names}
    allowed_names = all_names | log_names

    for i, expr_str in enumerate(form_exp_parts_str):
        try:
            # Extract all identifiers and create symbols for them
            identifiers = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", expr_str))
            local_dict = {name: Symbol(name) for name in identifiers}
            expr = parse_expr(expr_str.strip(), local_dict=local_dict)
        except Exception as e:
            return False, f"Failed to parse expression {i+1} ('{expr_str}'): {e}"

        # Check that all symbols in the expression are allowed
        for sym in expr.free_symbols:
            if sym.name not in allowed_names:
                return False, f"Unknown symbol '{sym.name}' in expression {i+1}. Allowed: {sorted(allowed_names)}"

    return True, None
