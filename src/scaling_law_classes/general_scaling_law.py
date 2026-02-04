import re
import sys
import heapq
import os.path
from functools import partial
from typing import Any, Dict, Callable, List, Tuple

import autograd.numpy as np
import pandas as pd
import torch
from torchmin import minimize, least_squares
from sympy import symbols, lambdify, parse_expr
from scipy.optimize import brentq

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.scaling_law import ScalingLaw


class PQItem(object):
    def __init__(self, loss, params):
        self.loss = loss
        self.params = params

    def __lt__(self, other):
        return self.loss > other.loss  # reversed because we want to retain lower loss params


def _infer_optim_params_names(form_exp_parts_str: List[str], param_names: List[str]) -> List[str]:
    """
    Infer which parameters are used in the form_exp_parts expressions and in what form.

    If a param appears as `logX`, add `logX` to optim_params_names (optimize in log space).
    If a param appears as `X` (not as part of logX), add `X` to optim_params_names.

    Returns sorted list of optim param names to ensure consistent ordering.
    """
    optim_params = set()
    log_param_names = {f"log{p}" for p in param_names}

    for expr in form_exp_parts_str:
        # Find all identifiers in the expression
        tokens = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr))

        for token in tokens:
            if token in log_param_names:
                # This is a log-space param (e.g., logA)
                optim_params.add(token)
            elif token in param_names:
                # This is an original param used directly (e.g., alpha, beta)
                optim_params.add(token)

    return sorted(optim_params)


class GeneralScalingLaw(ScalingLaw):
    """
    A general-purpose scaling law class that can represent any scaling law form.

    This class is designed to be subclassed (e.g., ChinchillaScalingLaw) or instantiated
    directly with custom form strings.

    Args:
        params: Dict mapping parameter names to values (e.g., {'A': 1.0, 'B': 2.0, ...})
        form_str: The loss formula as a string (e.g., "A / N**alpha + B / D**beta + E")
        params_str: Space-separated parameter names (e.g., "A B E alpha beta")
        vars_str: Space-separated variable names (e.g., "N D")
        form_exp_parts_str: List of log-space expressions for log-sum-exp computation
            (e.g., ["logA - alpha * logN", "logB - beta * logD", "logE"])
        is_fit_already: Whether this law has already been fit to data
    """

    def __init__(
        self,
        params: Dict[str, float],
        form_str: str,
        params_str: str,
        vars_str: str,
        form_exp_parts_str: List[str],
        is_fit_already: bool = False,
    ):
        super().__init__(params)
        self.is_fit_already = is_fit_already

        # Parse parameter names and create symbols
        param_symbols = symbols(params_str)
        self.param_names = sorted([p.strip() for p in params_str.split()])
        self.param_symbol_dict = {p: param_symbols[i] for i, p in enumerate(params_str.split())}
        self.params = {p: params.get(p) for p in self.param_names}

        # Parse variable names and create symbols
        var_symbols = symbols(vars_str)
        self.var_names = sorted([v.strip() for v in vars_str.split()])
        self.var_symbol_dict = {v: var_symbols[i] for i, v in enumerate(vars_str.split())}

        # Create log versions of parameters
        log_param_symbols = symbols(" ".join(f'log{p}' for p in self.param_names))
        self.log_params_names = [f'log{p}' for p in self.param_names]
        self.log_params_symbol_dict = {f'log{p}': log_param_symbols[i] for i, p in enumerate(self.param_names)}
        self.log_params = {f'log{p}': np.log(params.get(p)) if params.get(p) is not None else None
                          for p in self.param_names}

        # Create log versions of variables
        log_var_symbols = symbols(" ".join(f'log{v}' for v in self.var_names))
        self.log_vars_names = [f'log{v}' for v in self.var_names]
        self.log_vars_symbol_dict = {f'log{v}': log_var_symbols[i] for i, v in enumerate(self.var_names)}

        # Infer optim_params_names from form_exp_parts_str
        self.optim_params_names = _infer_optim_params_names(form_exp_parts_str, self.param_names)

        # Lambdify the main loss form
        self.form = lambdify(
            param_symbols + var_symbols,
            parse_expr(
                form_str.strip(),
                transformations="all",
                local_dict={**self.var_symbol_dict, **self.param_symbol_dict}
            ),
            "numpy"
        )

        # Lambdify each form_exp_parts expression
        self.form_exp_parts = []
        self.form_exp_parts_str = form_exp_parts_str
        for part in form_exp_parts_str:
            self.form_exp_parts.append(lambdify(
                param_symbols + var_symbols + log_param_symbols + log_var_symbols,
                parse_expr(
                    part.strip(),
                    transformations="all",
                    local_dict={
                        **self.param_symbol_dict, **self.var_symbol_dict,
                        **self.log_vars_symbol_dict, **self.log_params_symbol_dict
                    }
                ),
                "numpy"
            ))

    def apply_form_exp_parts(self, params_list: torch.Tensor, inps: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Evaluate the sympy expressions using the optimizer's current parameter values.

        Args:
            params_list: tensor of parameter values from optimizer, in order of self.optim_params_names
            inps: dict with variable tensors (e.g., 'N', 'D') and 'Loss' tensor
        """
        if not self.form_exp_parts:
            return []

        # Build dict from optimizer params
        opt_params = {}
        for i, name in enumerate(self.optim_params_names):
            opt_params[name] = params_list[i]

        # Compute all param values (both original and log versions)
        param_vals = {}
        log_param_vals = {}
        for p in self.param_names:
            log_name = f"log{p}"
            if log_name in opt_params:
                # This param is optimized in log space
                log_param_vals[log_name] = opt_params[log_name]
                param_vals[p] = torch.exp(opt_params[log_name])
            elif p in opt_params:
                # This param is optimized in original space
                param_vals[p] = opt_params[p]
                log_param_vals[log_name] = torch.log(opt_params[p])
            else:
                # This param is not being optimized, use stored value
                param_vals[p] = torch.tensor(self.params[p]) if self.params[p] is not None else torch.tensor(0.0)
                log_param_vals[log_name] = torch.log(param_vals[p])

        # Get variable values and compute log versions
        var_vals = {v: inps[v] for v in self.var_names}
        log_var_vals = {f"log{v}": torch.log(inps[v]) for v in self.var_names}

        # Build positional args in the correct order:
        # param_symbols + var_symbols + log_param_symbols + log_var_symbols
        # Note: symbols are created from the original (unsorted) order, but we sort param_names/var_names
        # We need to pass args in the order that matches how symbols were created
        original_param_order = self.param_symbol_dict.keys()
        original_var_order = self.var_symbol_dict.keys()

        args = []
        for p in original_param_order:
            args.append(param_vals[p])
        for v in original_var_order:
            args.append(var_vals[v])
        for p in original_param_order:
            args.append(log_param_vals[f"log{p}"])
        for v in original_var_order:
            args.append(log_var_vals[f"log{v}"])

        # Evaluate each expression
        lse_arr = [self.form_exp_parts[i](*args) for i in range(len(self.form_exp_parts))]

        # Convert scalars to tensors if needed
        lse_arr = [torch.tensor([l]) if not isinstance(l, torch.Tensor) else l for l in lse_arr]

        # Expand shape of all tensors to match the biggest one
        biggest_i = max(range(len(lse_arr)), key=lambda i: lse_arr[i].numel())
        return [lse_arr[i].expand_as(lse_arr[biggest_i]) for i in range(len(lse_arr))]

    def apply_form_exp_parts_numpy(self, params_list: np.ndarray, inps: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        NumPy version of apply_form_exp_parts for gradient-based optimization with autograd.
        """
        if not self.form_exp_parts:
            return []

        # Build dict from optimizer params
        opt_params = {}
        for i, name in enumerate(self.optim_params_names):
            opt_params[name] = params_list[i]

        # Compute all param values
        param_vals = {}
        log_param_vals = {}
        for p in self.param_names:
            log_name = f"log{p}"
            if log_name in opt_params:
                log_param_vals[log_name] = opt_params[log_name]
                param_vals[p] = np.exp(opt_params[log_name])
            elif p in opt_params:
                param_vals[p] = opt_params[p]
                log_param_vals[log_name] = np.log(opt_params[p])
            else:
                param_vals[p] = self.params[p] if self.params[p] is not None else 0.0
                log_param_vals[log_name] = np.log(param_vals[p])

        var_vals = {v: inps[v] for v in self.var_names}
        log_var_vals = {f"log{v}": np.log(inps[v]) for v in self.var_names}

        original_param_order = self.param_symbol_dict.keys()
        original_var_order = self.var_symbol_dict.keys()

        args = []
        for p in original_param_order:
            args.append(param_vals[p])
        for v in original_var_order:
            args.append(var_vals[v])
        for p in original_param_order:
            args.append(log_param_vals[f"log{p}"])
        for v in original_var_order:
            args.append(log_var_vals[f"log{v}"])

        lse_arr = [self.form_exp_parts[i](*args) for i in range(len(self.form_exp_parts))]

        # Ensure all are arrays and expand to match biggest shape
        lse_arr = [np.atleast_1d(l) for l in lse_arr]
        biggest_i = max(range(len(lse_arr)), key=lambda i: lse_arr[i].size)
        return [np.resize(lse_arr[i], lse_arr[biggest_i].shape) for i in range(len(lse_arr))]

    def loss_expr(self, *, N: float, D: float, **kwargs) -> float:
        """Compute loss given N and D using the lambdified form."""
        return self.form(**self.params, N=N, D=D)

    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        """
        Return D such that loss(N, D) == target_loss.

        Uses numerical root finding since the general form may not have an analytic solution.
        """
        def objective(D):
            return self.loss_expr(N=N, D=D, **other_vars) - target_loss

        # Search over a wide range of D values
        D_min, D_max = 1e3, 1e18

        # Check bounds
        loss_at_min = objective(D_min)
        loss_at_max = objective(D_max)

        if loss_at_min * loss_at_max > 0:
            raise ValueError(
                f"No solution found in range [{D_min}, {D_max}]. "
                f"Loss at D_min: {loss_at_min + target_loss}, Loss at D_max: {loss_at_max + target_loss}"
            )

        return brentq(objective, D_min, D_max, xtol=1e-10)

    def DL_to_N(self, D: float, L: float, **other_vars) -> float:
        """
        Return N such that loss(N, D) == L.

        Uses numerical root finding since the general form may not have an analytic solution.
        """
        def objective(N):
            return self.loss_expr(N=N, D=D, **other_vars) - L

        N_min, N_max = 1e3, 1e18

        loss_at_min = objective(N_min)
        loss_at_max = objective(N_max)

        if loss_at_min * loss_at_max > 0:
            raise ValueError(
                f"No solution found in range [{N_min}, {N_max}]. "
                f"Loss at N_min: {loss_at_min + L}, Loss at N_max: {loss_at_max + L}"
            )

        return brentq(objective, N_min, N_max, xtol=1e-10)

    def compute_optimal_train_tokens(self, x: float, T: float, L: float) -> float:
        """
        Generic implementation for finding optimal training tokens.

        This is a fallback that uses the loss equation directly. Subclasses may override
        with analytic solutions for better performance.

        For Chinchilla-like laws, this corresponds to Equation (12) in
        https://arxiv.org/pdf/2401.00448 - finding the optimal D given target loss L
        and inference tokens T.

        Returns the residual (should be zero at the optimal D).
        """
        # For general case, we need the derivative of loss w.r.t. D
        # This is a simplified version that assumes the standard form
        # Subclasses should override with their specific formula
        raise NotImplementedError(
            "compute_optimal_train_tokens must be overridden in subclasses with specific formulas. "
            "The general form requires knowing the derivative structure of the loss function."
        )

    def torch_loss(
        self,
        params_list: torch.Tensor,
        form_exp_parts: Callable,
        inp: Dict[str, torch.Tensor],
        tie_indices: List[List[int]] = [],
        loss_kwargs: Dict = {'loss_func': 'log_huber', 'delta': 1e-3},
    ) -> torch.Tensor:
        """Compute loss for optimization using PyTorch."""
        loss_func = loss_kwargs.get('loss_func', 'log_huber')
        delta = loss_kwargs.get('delta', 1e-3)

        for tie_params in tie_indices:
            tie_source = params_list[tie_params[0]]
            for i in tie_params[1:]:
                params_list[i] = tie_source

        pre = torch.stack(form_exp_parts(params_list, inp))
        post = torch.logsumexp(pre, dim=0)

        if loss_func == 'log_huber':
            return torch.nn.functional.huber_loss(
                post, torch.log(inp["Loss"]), delta=delta, reduction="none"
            ).sum()
        elif loss_func == 'huber':
            return torch.nn.functional.huber_loss(
                torch.exp(post), inp["Loss"], delta=delta, reduction="none"
            ).sum()
        elif loss_func == 'log_mae':
            return torch.abs(torch.log(inp["Loss"]) - post).sum()
        elif loss_func == 'log_mse':
            return ((torch.log(inp["Loss"]) - post) ** 2).sum()
        else:
            raise NotImplementedError(f"Loss function {loss_func} not implemented.")

    def numpy_loss(
        self,
        params_list: np.ndarray,
        form_exp_parts: Callable,
        inp: Dict[str, np.ndarray],
        tie_indices: List[List[int]] = [],
        loss_kwargs: Dict = {'loss_func': 'log_huber', 'delta': 1e-3},
    ) -> np.ndarray:
        """Compute loss for optimization using NumPy (compatible with autograd)."""
        loss_func = loss_kwargs.get('loss_func', 'log_huber')
        delta = loss_kwargs.get('delta', 1e-3)

        for tie_params in tie_indices:
            tie_source = params_list[tie_params[0]]
            for i in tie_params[1:]:
                params_list[i] = tie_source

        pre = np.stack(form_exp_parts(params_list, inp))
        from functools import reduce as functools_reduce
        post = functools_reduce(np.logaddexp, pre)

        if loss_func == 'log_huber':
            return np.sum(
                np.where(
                    np.abs(np.log(inp["Loss"]) - post) <= delta,
                    0.5 * (np.log(inp["Loss"]) - post)**2,
                    delta * (np.abs(np.log(inp["Loss"]) - post) - 0.5 * delta)))
        elif loss_func == 'huber':
            return np.sum(
                np.where(
                    np.abs(inp["Loss"] - np.exp(post)) <= delta,
                    0.5 * (inp["Loss"] - np.exp(post))**2,
                    delta * (np.abs(inp["Loss"] - np.exp(post)) - 0.5 * delta)))
        elif loss_func == 'log_mae':
            return np.abs(np.log(inp["Loss"]) - post).sum()
        elif loss_func == 'log_mse':
            return ((np.log(inp["Loss"]) - post) ** 2).sum()
        else:
            raise NotImplementedError(f"Loss function {loss_func} not implemented.")

    def iso_loss_function(self, target_loss: float, **other_vars):
        return super().iso_loss_function(target_loss, **other_vars)

    def compute_optimal_allocation(self, C, **kw):
        return super().compute_optimal_allocation(C, **kw)

    def fit(self, data, init_params: Dict[str, float] | None = None, *args, **kwargs):
        """
        Fit the scaling law to data.

        Args:
            data: DataFrame or path to CSV file with columns for each variable plus 'Loss'
            init_params: Optional dict of initial parameter values for optimization
        """
        if self.is_fit_already:
            raise RuntimeError("Scaling Law is already fit.")

        # Handle both DataFrame and file path inputs
        if isinstance(data, str):
            if not os.path.isfile(data):
                raise FileNotFoundError(f"Data source file {data} not found.")
            if not data.endswith('.csv'):
                raise ValueError("Data source file must be a CSV file.")
            data = pd.read_csv(data)

        # Check required columns
        missing_cols = [v for v in self.var_names if v not in data.columns]
        if missing_cols:
            raise ValueError(f"Data must contain columns: {missing_cols}")
        if "Loss" not in data.columns:
            raise ValueError("Data must contain 'Loss' column.")

        # Build input tensors
        inp_torch = {v: torch.tensor(data[v].values.astype(float), dtype=torch.float32)
                     for v in self.var_names}
        inp_torch["Loss"] = torch.tensor(data["Loss"].values.astype(float), dtype=torch.float32)

        # Build grid for optimization
        if init_params is not None:
            grid = {}
            for name in self.optim_params_names:
                if name.startswith('log'):
                    base_name = name[3:]
                    grid[name] = torch.tensor([np.log(init_params[base_name])], dtype=torch.float32)
                else:
                    grid[name] = torch.tensor([init_params[name]], dtype=torch.float32)
        else:
            # Default grid - works for Chinchilla-like laws
            grid = {}
            for name in self.optim_params_names:
                if name.startswith('log'):
                    grid[name] = torch.arange(start=0, end=25+5, step=5.0)
                elif name in ('alpha', 'beta'):
                    grid[name] = torch.arange(start=0, end=2+0.5, step=0.5)
                else:
                    grid[name] = torch.arange(start=-1, end=1+0.5, step=0.5)

        loss, theta, _pq = minimize_scl_loss(
            init_params=None,
            grid_specs=grid,
            torch_loss=self.torch_loss,
            form_exp_parts=self.apply_form_exp_parts,
            inp_torch=inp_torch,
            loss_kwargs={"tie_groups": kwargs.get('tie', []), "delta": 1e-3, "loss_func": "log_huber"},
            param_names=self.optim_params_names,
        )

        # Convert optimizer params back to original param names
        fit_params = {}
        for name in self.optim_params_names:
            if name.startswith('log'):
                base_name = name[3:]
                fit_params[base_name] = np.exp(theta[name])
            else:
                fit_params[name] = theta[name]

        return loss, fit_params


def minimize_scl_loss(
    init_params: List[float],
    grid_specs: Dict[str, np.ndarray],
    torch_loss: Callable,
    form_exp_parts: Callable,
    inp_torch: Dict[str, torch.Tensor],
    param_names: List[str] = [],
    loss_kwargs: Dict[str, Any] = None,
    method: str = 'BFGS',
    max_opt_inits: int = -1,
    keep_best_k_from_init_grid: int = -1,
    tol: float = None,
) -> Tuple[float, Dict[str, float], List[PQItem]]:
    """
    Minimize scaling law loss over a grid of initial parameter values.

    From Hoffmann et al: Uses LBFGS algorithm to find local minima, started on a grid
    of initializations.

    Args:
        init_params: Initial parameter values (ignored if grid_specs provided)
        grid_specs: Dict mapping param names to arrays of initial values
        torch_loss: Loss function to minimize
        form_exp_parts: Function that computes log-sum-exp parts
        inp_torch: Input tensors (N, D, Loss, etc.)
        param_names: Names of parameters being optimized
        loss_kwargs: Additional kwargs for loss function
        method: Optimization method ('BFGS', 'grid', etc.)
        max_opt_inits: Maximum number of initializations to try (-1 = no limit)
        keep_best_k_from_init_grid: Only optimize from top k initial points
        tol: Optimization tolerance

    Returns:
        Tuple of (best_loss, best_params_dict, priority_queue_of_results)
    """
    if loss_kwargs is None:
        loss_kwargs = {}

    best_loss = np.inf
    best_params = None
    pq = []
    i = 0

    loss_kwargs['tie_indices'] = [
        [param_names.index(p) if p in param_names else param_names.index(p[3:])
         for p in tie_group]
        for tie_group in loss_kwargs.get('tie_groups', [])
    ]

    # Build grid of initial parameters
    grid = torch.stack(torch.meshgrid(
        *[grid_specs[key] if key in grid_specs else grid_specs[f"log{key}"]
          for key in param_names],
        indexing='ij'
    ))
    grid = grid.permute(*torch.arange(grid.ndim - 1, -1, -1)).reshape(-1, len(grid_specs))
    grid = grid[torch.randperm(grid.size(0))]

    if keep_best_k_from_init_grid > 0:
        init_pq = []
        for init_p in grid:
            init_loss = torch_loss(init_p, form_exp_parts, inp_torch, loss_kwargs=loss_kwargs)
            if len(init_pq) < keep_best_k_from_init_grid:
                heapq.heappush(init_pq, PQItem(init_loss, init_p))
            elif init_loss < init_pq[0].loss:
                heapq.heappushpop(init_pq, PQItem(init_loss, init_p))
        grid = [pq_item.params for pq_item in heapq.nlargest(keep_best_k_from_init_grid, init_pq)]

    for init_p in grid:
        if method == 'grid':
            params = init_p
            loss = torch_loss(init_p, form_exp_parts, inp_torch, loss_kwargs=loss_kwargs)
            success = True
        else:
            obj = partial(torch_loss, form_exp_parts=form_exp_parts, inp=inp_torch, loss_kwargs=loss_kwargs)
            if method == 'nonlinear_least_squares':
                result = least_squares(obj, init_p)
            else:
                result = minimize(obj, init_p, tol=tol, method=method)

            for tie_params in loss_kwargs.get('tie_indices', []):
                tie_source = result.x[tie_params[0]]
                for idx in tie_params[1:]:
                    result.x[idx] = tie_source
            params, loss, success = result.x, result.fun, result.success

        if success and loss < best_loss:
            best_loss = loss
            best_params = params

        if len(pq) < 100:
            heapq.heappush(pq, PQItem(loss, params))
        elif loss < pq[0].loss:
            heapq.heappushpop(pq, PQItem(loss, params))

        i += 1
        if i == max_opt_inits:
            break

    if best_params is not None:
        best_params_dict = {param_names[i]: float(best_params[i]) for i in range(len(param_names))}
    else:
        best_params_dict = None

    return best_loss, best_params_dict, pq
