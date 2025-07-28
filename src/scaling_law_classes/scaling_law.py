from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import brentq, minimize_scalar, newton
from sklearn.metrics import r2_score


@dataclass(frozen=True)
class LawParams:
    A: float
    B: float
    irreducible: float
    alpha: float
    beta: float
    extras: Mapping[str, float] = field(default_factory=dict)


class ScalingLaw(ABC):
    """Common API for all scaling‑law families.

    Every ScalingLaw offers the following abilities:

    - Fit that ScalingLaw with data.
    - Predict L from D,N,vars.
    - Estimate D*,N* from C (or a train/inference breakdown of it)
    - Estimate D_opt / N_opt ratio

    - Produce an iso-loss curve across (N,D) options


    """

    # --- model‑specific metadata -----------------------------------------
    variables: Sequence[str] = ()
    default_vars: Mapping[str, float] = {}
    FLOPS_COEFF: float = 6.0  # override if k ≠ 6 for your law

    def __init__(self, params: LawParams):
        self.params = params
        try:
            self.G = ((params.alpha * params.A) / (params.beta * params.B)) ** (
                1 / (params.alpha + params.beta)
            )
        except Exception:
            self.G = None

    def set_flops_coeff(self, flops_coeff):
        self.FLOPS_COEFF = flops_coeff

    def flops(self, *, N: float, D: float, **vars) -> float:
        """Default: k·N·D with a class-level constant k."""
        return self.FLOPS_COEFF * N * D

    # ---------------- mathematics ---------------------------------------
    @abstractmethod
    def loss_expr(self, **vars: float): ...

    @staticmethod
    @abstractmethod
    def torch_loss(inp: torch.Tensor, theta: torch.Tensor, **kw): ...

    @staticmethod
    @abstractmethod
    def numpy_loss(X: np.ndarray, *theta, **kw) -> np.ndarray: ...

    def D_to_N(self, D):
        return (D * self.G) ** (self.beta / self.alpha) * self.G

    # ---------------- general iso‑loss utilities -------------------------
    # Each subclass *must* supply an analytic or numeric implementation.
    @abstractmethod
    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        """Return *D* such that loss(N,D,**other_vars) == target_loss.

        Subclasses should raise **ValueError** if the requested (N, target_loss)
        combination is infeasible for any *D > 0* given *other_vars*.
        """

    @abstractmethod
    def DL_to_N(self, D, L):
        """
        Minimum number of model params needed to reach L model loss after D tokens.
        """

    def get_parameters(self):
        return self.params

    # ---------------- convenience --------------------------------------
    def loss(self, **vars):
        merged = {**self.default_vars, **vars}
        missing = [v for v in self.variables if v not in merged]
        if missing:
            raise ValueError(f"Missing vars {missing}")

        result = self.loss_expr(**merged)

        # Only convert to float if it's a scalar
        if hasattr(result, "__len__") and len(result) > 1:
            return result  # Return array as-is
        else:
            return float(result)  # Convert scalar to float

    # def flops(self, **vars):
    #     if {"N", "D"}.issubset(self.variables):
    #         return 6.0 * vars["N"] * vars["D"]
    #     raise NotImplementedError("Override flops() for non N,D laws")

    def evaluate_loss(self, x, y_true):
        # print(*x)
        y_pred = self.loss(**x)
        return r2_score(y_true, y_pred)

    def iso_compute_function(self, C: float, **other_vars):
        """
        Return a callable  f(N) → D  such that  FLOPs(N,D)=C.

        If flops() is the simple k·N·D form (most laws) we give an
        *analytic* solution; otherwise we fall back to a 1-D root-finder.
        """
        k = self.FLOPS_COEFF

        # Fast analytic path ──────────────
        if self.flops(N=1.0, D=1.0, **other_vars) == k:
            return lambda N: C / (k * N)

        # Generic numeric path ───────────
        def _N_to_D(N: float) -> float:
            def fn(D):
                return self.flops(N=N, D=D, **other_vars) - C
            return brentq(fn, 1e-15, 1e15, maxiter=256)

        return _N_to_D

    def iso_loss_function(
        self, target_loss: float, **other_vars
    ) -> Callable[[float], float]:
        """Return a callable *f(N) → D* that yields the data size *D* needed
        to stay on the **L = target_loss** iso‑loss contour.

        Parameters
        ----------
        target_loss   : Desired loss level. Must be **> irreducible loss**.
        **other_vars  : Values for variables *other than* N and D that are
                        required by the specific scaling law (e.g. *K* or *U*).

        Notes
        -----
        The returned function closes over *target_loss* and *other_vars*; it
        performs the analytic/numeric computation every time it is called so
        that vectorisation (via `np.vectorize`) is still possible if needed.
        """
        # We deliberately capture a *copy* of other_vars to avoid accidental
        # mutation by client code after creating the callable.
        extra = dict(other_vars)

        def _N_to_D(N: float) -> float:
            return self.N_to_D(N, target_loss, **extra)

        return _N_to_D

    def compute_optimal_allocation(
        self,
        C: float,
        N_bounds: tuple[float, float] = (1e3, 1e12),
        **other_vars,
    ):
        """
        Minimise loss(N,D,… ) subject to FLOPs = C.

        Works for *any* concrete law because we only ever call
        `loss()` and `flops()`.
        """
        # Construct D(N) along the iso-compute curve
        D_of_N = self.iso_compute_function(C, **other_vars)

        # Objective:   f(N) = loss(N, D(N), …)
        def obj(N: float) -> float:
            return self.loss(N=N, D=D_of_N(N), **other_vars)

        # 1-D search along N .  Brent is robust & scalar-only.
        res = minimize_scalar(obj, bounds=N_bounds, method="bounded")
        if not res.success:
            raise RuntimeError("optimise_under_compute: minimise_scalar failed")

        N_opt = res.x
        D_opt = D_of_N(N_opt)
        return dict(
            loss=obj(N_opt),
            model=N_opt,
            data=D_opt,
            flops=C,
            # pass through any extra vars so the caller sees what mattered
            **other_vars,
        )

    def compute_optimal_allocation_inference(
        self, compute_budget, D_inference, **kwargs
    ):
        """
        Compute optimal D, N given compute budget and inference tokens

        See https://arxiv.org/abs/2401.00448
        """
        if "unique_data" in kwargs:
            # To make this work for Scaling DC LMs, we need to derive Equation (12) from
            # https://arxiv.org/pdf/2401.00448 for the Scaling DC LMs equation
            raise NotImplementedError(
                "Optimal inference allocation for data-constrained laws not implemented."
            )

        # Allow also passing in target loss directly
        loss = kwargs.get(
            "loss", self.compute_optimal_allocation(compute_budget)["loss"]
        )

        # Use Newton's method to find optimal training tokens
        D_opt = newton(
            self.compute_optimal_train_tokens,
            1e8,  # Initial guess
            tol=1e-5,
            maxiter=100,
            rtol=1e-8,
            args=(D_inference, loss),
            disp=False,  # Set to True for debugging
        )

        approximation_error = abs(
            self.compute_optimal_train_tokens(D_opt, D_inference, loss)
        )
        if approximation_error > 1e-6:
            raise RuntimeError(
                f"Could not find optimal training tokens. Error: {approximation_error}."
            )

        N_opt = self.DL_to_N(D_opt, loss)
        flops_train = self.flops(N=N_opt, D=D_opt)
        flops_inference = self.flops(N=N_opt, D=D_inference)

        return {
            "loss": loss,
            "data": D_opt,
            "model": N_opt,
            "flops_train": flops_train,
            "flops_inference": flops_inference,
            "flops_total": flops_train + flops_inference,
        }

    @abstractmethod
    def compute_optimal_train_tokens(self, x, T, L):
        """
        Equation (12) in https://arxiv.org/pdf/2401.00448

        Find the optimal number of tokens (D) to train on for a model
        of quality L (pre-training loss L) and run inference for T tokens.
        This method is used by a solver (e.g. Newton's method) to find root (optimal D).
        We cannot use a formula because there is no analytical formula when T > 0.

        The equation is:
        (β·B/α + B)·D^(-β) + (T·β·B)/(3·α)·D^(-β-1) + E - L = 0
        """


@dataclass(frozen=True)
class ScalingLawWrapper:
    name: str
    scaling_law: ScalingLaw
    paper: str
    publication_date: str
    model_architecture: str
    training_data: str
    languages: List[str]
    compute_budget_range: Tuple[int, int]
    extra_args: List[str]
    notes: str
