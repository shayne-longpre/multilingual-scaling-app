import sys
import math
from typing import Callable, Dict, Iterable, List
import numpy as np
import torch
from scipy.optimize import brentq

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.scaling_law import ScalingLaw
from src.scaling_law_classes.basic_scaling_law import BasicScalingLaw


class DataConstrainedScalingLaw(ScalingLaw):
    default_vars = {"N": 1.0, "D": 1.0, "U": 1.0}

    # Param name mappings for optimizer
    optim_params_names = ['logA', 'logB', 'logE', 'alpha', 'beta', 'rd_star', 'rn_star']

    def form_exp_parts(self, params_list: torch.Tensor, inp: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Evaluate the log-sum-exp parts using the optimizer's current parameter values.

        Args:
            params_list: tensor of parameter values from optimizer [logA, logB, logE, alpha, beta, rd_star, rn_star]
            inp: dict with 'UN', 'U', 'RD', 'RN', 'Loss' tensors
        """
        logA, logB, logE, alpha, beta, rd_star, rn_star = (
            params_list[0], params_list[1], params_list[2],
            params_list[3], params_list[4], params_list[5], params_list[6]
        )
        UN = inp["UN"]
        U = inp["U"]
        RD = inp["RD"]
        RN = inp["RN"]
        tm = UN + UN * rn_star * (1 - torch.exp(-RN / rn_star))
        td = U + U * rd_star * (1 - torch.exp(-RD / rd_star))
        return [
            logA - alpha * torch.log(tm),
            logB - beta * torch.log(td),
            logE.expand(inp["Loss"].shape[0]),
        ]

    # Alias for consistency with other scaling law classes
    apply_form_exp_parts = form_exp_parts

    # --- NumPy loss ------------------------------------------------------
    def loss_expr(self, *, N: float, D: float, U: float, **kwargs):
        if U > D:
            U = D

        p = self.params

        RD = np.maximum((D / U) - 1, 0)
        UN = np.minimum(N, self.D_to_N(U))
        RN = np.maximum((N / UN) - 1, 0)
        model_denom = UN + UN * p['rn_star'] * (
            1 - np.exp(-1 * RN / p['rn_star'])
        )
        data_denom = U + U * p['rd_star'] * (
            1 - np.exp(-1 * RD / p['rd_star'])
        )

        loss = (
            p['E']
            + (p['A'] / (model_denom**p['alpha']))
            + (p['B'] / (data_denom**p['beta']))
        )
        return loss

    def DL_to_N(self, D, L):
        return 0.0  # TODO

    def compute_optimal_train_tokens(self, x, T, L):
        return 0.0 # TODO

    # --- Numeric N → D using root‑finder --------------------------------
    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        U = other_vars.get("U")
        if U is None:
            raise ValueError("DataConstrainedScalingLaw.N_to_D requires keyword 'U'")

        # Quick feasibility check – use current loss at some D to bracket root
        def loss_minus_L(D):
            return self.loss(N=N, D=D, U=U) - target_loss

        # Attempt to find a sign change automatically.
        D_low, D_high = 1e-6, 1e6
        try:
            # Expand upper bound until the function becomes positive
            while loss_minus_L(D_high) > 0 and D_high < 1e12:
                D_high *= 10
            # Ensure lower bound is below target
            while loss_minus_L(D_low) < 0 and D_low > 1e-12:
                D_low /= 10
            root = brentq(loss_minus_L, D_low, D_high, maxiter=256)
            return root
        except ValueError as e:
            raise ValueError("Unable to bracket iso‑loss root for given N.") from e

    def torch_loss(
        self,
        params_list: torch.Tensor,
        form_exp_parts: Callable[[List[float], Dict[str, torch.Tensor]], List[torch.Tensor]],
        inp: Dict[str, torch.Tensor],
        tie_indices: List[List[int]] = [],
        loss_kwargs: Dict = {'loss_func': 'log_huber', 'delta': 1e-3}
    ) -> torch.Tensor:
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
        form_exp_parts: Callable[[List[float], Dict[str, np.ndarray]], List[np.ndarray]],
        inp: Dict[str, np.ndarray],
        tie_indices: List[List[int]] = [],
        loss_kwargs: Dict = {'loss_func': 'log_huber', 'delta': 1e-3},
    ) -> np.ndarray:
        loss_func = loss_kwargs.get('loss_func', 'log_huber')
        delta = loss_kwargs.get('delta', 1e-3)

        for tie_params in tie_indices:
            tie_source = params_list[tie_params[0]]
            for i in tie_params[1:]:
                params_list[i] = tie_source
        pre = np.stack(form_exp_parts(params_list, **inp))
        post = np.logaddexp.reduce(pre, axis=0)

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
        if "U" not in other_vars:
            raise ValueError("iso_loss_function requires keyword argument 'U'")
        return super().iso_loss_function(target_loss, **other_vars)

    def compute_optimal_allocation(self, C, *, U, **kw):
        return super().compute_optimal_allocation(C, U=U, **kw)

    def fit(self, data, *args, **kwargs):
        from src.scaling_law_classes.general_scaling_law import minimize_scl_loss

        unique_tokens = data["U"].max()
        pre_epoch_sample = data[data["D"] <= unique_tokens]

        min_epochs = round(pre_epoch_sample["D"].min() / unique_tokens, 2)
        max_epochs = round(pre_epoch_sample["D"].max() / unique_tokens, 2)
        print(f"Number of data samples <1 epoch: {len(pre_epoch_sample)} / {len(data)}. Ranging from {min_epochs} to {max_epochs} epochs.")

        # Fit BasicScalingLaw to pre-epoch data to get initial parameters
        basic_law = BasicScalingLaw(params={
            'A': 1.0, 'B': 1.0, 'E': 1.0, 'alpha': 0.5, 'beta': 0.5
        })
        orig_loss, p0 = basic_law.fit(pre_epoch_sample, **kwargs)
        a0, b0, e0 = map(math.log, [p0['A'], p0['B'], p0['E']])
        print(f"BasicScalingLaw fit loss: {orig_loss}, params: {p0}")

        alpha, beta = p0['alpha'], p0['beta']

        # Compute G ratio for N_sat calculation (G = (A/B)^(1/(alpha+beta)) * (beta/alpha)^(beta/(alpha+beta)))
        G = (p0['A'] / p0['B']) ** (1 / (alpha + beta)) * (beta / alpha) ** (beta / (alpha + beta))

        def row_vec(r):
            # N_sat – maximum model size that can be trained effectively with U tokens
            N_sat = (unique_tokens * G) ** (beta / alpha) * G

            UN = min(r["N"], N_sat)                    # model denominator base
            RD = max(r["D"] / unique_tokens - 1, 0)    # data reuse
            RN = max(r["N"] / UN - 1, 0) if UN > 0 else 0  # model reuse

            return [UN, unique_tokens, RD, RN]

        X = np.stack([row_vec(r) for _, r in data.iterrows()]).astype(float)
        y = data["Loss"].values.astype(float)

        post_epoch_sample = data[data["D"] >= unique_tokens]
        if len(post_epoch_sample) > 0:
            min_epochs_post = round(post_epoch_sample["D"].min() / unique_tokens, 2)
            max_epochs_post = round(post_epoch_sample["D"].max() / unique_tokens, 2)
            print(f"Number of data samples >1 epoch: {len(post_epoch_sample)} / {len(data)}. Ranging from {min_epochs_post} to {max_epochs_post} epochs.")

        # Create input dict with tensors
        inp_torch = {
            "UN": torch.tensor(X[:, 0], dtype=torch.float32),
            "U": torch.tensor(X[:, 1], dtype=torch.float32),
            "RD": torch.tensor(X[:, 2], dtype=torch.float32),
            "RN": torch.tensor(X[:, 3], dtype=torch.float32),
            "Loss": torch.tensor(y, dtype=torch.float32),
        }

        # Grid search over rd_star and rn_star, keeping other params fixed from BasicScalingLaw fit
        grid = {
            'logA': torch.tensor([a0]),
            'logB': torch.tensor([b0]),
            'logE': torch.tensor([e0]),
            'alpha': torch.tensor([alpha]),
            'beta': torch.tensor([beta]),
            'rd_star': torch.arange(start=0.1, end=20.1, step=2.0),
            'rn_star': torch.arange(start=0.1, end=20.1, step=2.0),
        }

        loss, theta, _pq = minimize_scl_loss(
            init_params     = None,  # ignored because grid_specs is provided
            grid_specs      = grid,
            torch_loss      = self.torch_loss,
            form_exp_parts  = self.apply_form_exp_parts,
            inp_torch       = inp_torch,
            loss_kwargs     = {"tie_groups": kwargs.get('tie', []), "delta": 1e-3, "loss_func": "log_huber"},
            param_names     = self.optim_params_names,
        )

        # theta is a dict with optim_params_names keys
        # Convert to fit_params with original param names
        fit_params = {
            "A": np.exp(theta['logA']),
            "B": np.exp(theta['logB']),
            "E": np.exp(theta['logE']),
            "alpha": theta['alpha'],
            "beta": theta['beta'],
            "rd_star": theta['rd_star'],
            "rn_star": theta['rn_star'],
        }
        return loss, fit_params
