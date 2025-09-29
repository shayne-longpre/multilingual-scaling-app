import sys
import heapq
from functools import partial
from typing import Any, NamedTuple, Dict, Callable, List
# import numpy as np
import autograd.numpy as np
from autograd.scipy.stats import norm
import torch
from torchmin import minimize, least_squares
# from scipy.optimize import brentq, curve_fit, minimize, least_squares, OptimizeWarning

sys.path.append("./")
sys.path.append("src/")

from src.scaling_law_classes.scaling_law import ScalingLaw, LawParams



def huber_logpdf(x, delta=1e-3, loc=0, scale=1):
    x = (x-loc)/scale
    cond = np.abs(x) <= delta
    loss = np.where(cond, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))
    huber_normalizing_factor = np.sqrt(2*np.pi) * (1 - 2*norm.sf(delta)) + 2 * np.exp(-0.5*delta**2)/delta
    return -loss - np.log(huber_normalizing_factor) - np.log(scale)

def huber_pdf(x, delta=1e-3, loc=0, scale=1):
    return np.exp(huber_logpdf(x, delta=delta, loc=loc, scale=scale))

# Define the objective function to be minimized
def scaled_log_huber_objective(predictions, losses, sigma=1e-1, delta=1e-3):
    return -np.sum(huber_logpdf(np.log(losses), loc=np.log(predictions), scale=np.exp(sigma), delta=delta))

def log_huber_loss_objective(predictions, losses, delta=1e-3):
    # Calculate the difference
    diff = np.log(losses) - np.log(predictions)
    # Calculate the condition for Huber loss
    cond = np.abs(diff) <= delta
    # Apply Huber loss formula
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return np.sum(loss)

def log_mae_objective(predictions, losses):
    return np.sum(np.abs(np.log(losses) - predictions))

def log_mse_objective(predictions, losses):
    return np.sum((np.log(losses) - predictions)**2)

def huber_loss_objective(predictions, losses, delta=1e-3):
    diff = losses - predictions
    # Calculate the condition for Huber loss
    cond = np.abs(diff) <= delta
    # Apply Huber loss formula
    loss = np.where(cond, 0.5 * diff**2, delta * (np.abs(diff) - 0.5 * delta))
    return np.sum(loss)


# transform parameters back from log space
def untransform_params(param_array):
    if len(np.shape(param_array)) == 2:
        return np.hstack((np.exp(param_array[:, :3]), param_array[:, 3:]))
    else:
        return np.hstack((np.exp(param_array[:3]), param_array[3:]))


class PQItem(object):
    def __init__(self, loss, params):
        self.loss = loss
        self.params = params

    def __lt__(self, other):
        return self.loss > other.loss # reversed because we want to retain lower loss params



class ScalingLaw(ScalingLaw):
    variables = ("N", "D", )
    default_vars = {"N": 1.0, "D": 1.0}

    def __init__(self, params: LawParams, form_str=None, params_str=None):#, variables=None):
        super().__init__(params)
        self.form
        # try:
        #     self.G = ((params.alpha * params.A) / (params.beta * params.B)) ** (
        #         1 / (params.alpha + params.beta)
        #     )
        # except Exception:
        #     self.G = None

        from sympy import symbols, lambdify, parse_expr
        params = symbols(params_str)
        param_dict = {}
        for i, param in enumerate(params_str.split(',')):
            param_dict[param.strip()] = params[i]
        self.f = lambdify(params, parse_expr(form_str, transformations="all"), 'numpy')


    # Define the log-sum-exp function
    def form_exp_parts(params: Dict, inps: Dict):
        return [
            params['a'] - params['alpha'] * torch.log(inps['N']),
            params['b'] - params['beta'] * torch.log(inps['D']),
            params['e'].expand(inps['D'].shape[0])
        ]    
    # def form_exp_parts(params: Dict, inp: torch.Tensor):
    #     return [
    #         params['a'] - params['alpha'] * torch.log(inp[:, 0]),
    #         params['b'] - params['beta'] * torch.log(inp[:, 1]),
    #         params['e'].expand(inp.shape[0])
    #     ]
    

    # --- NumPy loss ------------------------------------------------------
    def loss_expr(self, *, N: float, D: float, U: float, **kwargs):
        p = self.params
        return self.form(p.A, p.B, p.irreducible, p.alpha, p.beta, N, D)
    
    # --- Analytic N → D on iso‑loss ------------------------------------
    def N_to_D(self, N: float, target_loss: float, **other_vars) -> float:
        p = self.params
        L_eff = target_loss - p.irreducible
        if L_eff <= 0:
            raise ValueError("target_loss must exceed irreducible loss")
        denom = L_eff - p.A / N**p.alpha
        if denom <= 0:
            raise ValueError(
                "No finite D can satisfy the loss at this N (denominator ≤ 0)"
            )
        D = (p.B / denom) ** (1.0 / p.beta)
        return D

    def DL_to_N(self, D, L):
        """
        Minimum number of model params needed to reach L model loss after D tokens.

        This is the regular Chinchilla equation solved for N.
        """
        p = self.params
        L_eff = L - p.irreducible

        if L_eff <= 0:
            raise ValueError(
                f"Target loss {L} must exceed irreducible loss {p.irreducible}"
            )

        denominator = L_eff - p.B / (D**p.beta)

        if denominator <= 0:
            raise ValueError(
                f"Cannot achieve loss {L} with {D} tokens - need more data"
            )

        partial_result = p.A / denominator
        return partial_result ** (1 / p.alpha)

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
        p = self.params

        coeff_1 = (p.beta * p.B) / p.alpha + p.B
        coeff_2 = (T * p.beta * p.B) / (3 * p.alpha)
        loss_diff = p.irreducible - L

        return (
            coeff_1 * x ** (-1 * p.beta)
            + coeff_2 * x ** ((-1 * p.beta) - 1)
            + loss_diff
        )


    @staticmethod
    def torch_loss(form_exp_parts: Callable[[Dict, torch.Tensor], List[torch.Tensor]], inp: Dict, params: Dict, loss_func: str = 'log_huber', tie: bool = False, delta: float = 1e-3) -> torch.Tensor:
        pre = torch.stack(form_exp_parts(params, inp))
        post = torch.logsumexp(pre, dim=0) # log scale

        if loss_func == 'log_huber':
            return torch.nn.functional.huber_loss(
                post, torch.log(inp["Loss"]), delta=delta
            ).sum()
        elif loss_func == 'huber':
            return torch.nn.functional.huber_loss(
                torch.exp(post), inp["Loss"], delta=delta
            ).sum()
        # elif loss_func == 'scaled_log_huber': # NOT WORKING YET
        #     return torch.nn.functional.huber_loss(
        #         post, torch.log(inp[:, 2]), delta=delta
        #     ).sum() / torch.exp(params['sigma']) + 0.5 * torch.log(2 * torch.pi) + params['sigma'] + torch.log(
        #         torch.sqrt(2 * torch.pi) * (1 - 2 * torch.exp(-0.5 * (delta)**2) * norm.sf(delta)) + 2 * torch.exp(-0.5 * (delta)**2) / delta
        #     ) # p.sqrt(2*np.pi) * (1 - 2*norm.sf(delta)) + 2 * np.exp(-0.5*delta**2)/delta  + np.log(scale)
        elif loss_func == 'log_mae':
            return torch.abs(torch.log(inp["Loss"]) - post).sum()
        elif loss_func == 'log_mse':
            return ((torch.log(inp["Loss"]) - post) ** 2).sum()
        else:
            raise NotImplementedError(f"Loss function {loss_func} not implemented.")
    
    @staticmethod
    def numpy_loss(inp: np.ndarray, params: np.ndarray) -> np.ndarray:
        a, b, e, alpha, beta = params
        N, D = inp[:, 0], inp[:, 1]
        return np.exp(e) + np.exp(a) / N**alpha + np.exp(b) / D**beta

    def iso_loss_function(self, target_loss: float, **other_vars):
        if "U" not in other_vars:
            raise ValueError("iso_loss_function requires keyword argument 'U'")
        return super().iso_loss_function(target_loss, **other_vars)

    def compute_optimal_allocation(self, C, *, U, **kw):
        return super().compute_optimal_allocation(C, U=U, **kw)

    @classmethod
    def fit(cls, data, *args, **kw):
        
        # x_nd = data[["N", "D"]].values.astype(float)
        N  = data["N"].values.astype(float)
        D  = data["D"].values.astype(float)
        model_losses    = data["Loss"].values.astype(float)
        print(f"Data points: {len(data)}.")


    def minimize_scl_loss(
            init_params: List[float],
            grid_specs: Dict[str, np.ndarray],
            torch_loss: Callable[[Callable, torch.Tensor, Dict, str, bool, float], torch.Tensor],
            inp_torch: torch.Tensor,
            loss_kwargs: Dict[str, Any] = None,
        ):
        indices=None,
        init_grid=None,
        keep_best_k_from_init_grid=-1,
        method='BFGS',
        use_grad=False,
        tol=None,
        add_sigma=False,
        max_opt_inits=-1, #no max by default
        dense_init=False,
        """
        From hoffman, et al:
        We use the LBFGS algorithm to find local minima of the objective above, started on a grid
        of initialisation given by:
        𝛼 ∈ {0., 0.5, . . . , 2.},
        𝛽 ∈ {0., 0.5, . . . , 2.},
        𝑒 ∈ {−1., −.5, . . . , 1.},
        𝑎 ∈ {0, 5, . . . , 25}, and
        𝑏 ∈ {0, 5, . . . , 25}
        """

        best_loss = np.inf
        best_params = None
        pq = []
        param_list = []
        i = 0
        if init_params:
            assert grid_specs is None, "Cannot provide both init_params and grid_specs"
            grid_specs = [init_params]
        if grid_specs is None:
            grid_specs = {
                'a': np.arange(0, 25, 5),
                'b': np.arange(0, 25, 5),
                'e': np.arange(-1, 1, 0.5),
                'alpha': np.arange(0, 2, 0.5),
                'beta': np.arange(0, 2, 0.5)
            }
        # if tie_alpha_beta:
        #     grid_specs['beta'] = [0] # dummy value, will be set to alpha later

        grid = np.array(np.meshgrid(
            *[grid_specs[key] for key in sorted(grid_specs.keys())]
        )).T.reshape(-1, len(grid_specs))
        np.random.shuffle(grid)

        if keep_best_k_from_init_grid > 0:
            init_pq = []
            for init_params in grid:
                init_loss = torch_loss(init_params, N[indices], D[indices], model_losses[indices])#, form=form)
                if len(init_pq) < keep_best_k_from_init_grid:
                    heapq.heappush(init_pq, PQItem(init_loss, init_params))
                elif init_loss < init_pq[0].loss:
                    heapq.heappushpop(init_pq, PQItem(init_loss, init_params))
            grid = [pq_item.params for pq_item in heapq.nlargest(keep_best_k_from_init_grid, init_pq)]

        results_dict = {}
        for init_params in grid:
            if method == 'grid':
                params = init_params
                loss = torch_loss(init_params, inp_torch, loss_kwargs=loss_kwargs)
                success = True
            else:
                # if add_sigma or obj_name in ['scaled_log_huber']:
                #     init_params = init_params + [0]
                if method == 'nonlinear_least_squares':
                    result = least_squares(obj, init_params, args=(N[indices], D[indices], model_losses[indices]), )
                else:
                    result = minimize(obj, init_params, args=(N[indices], D[indices], model_losses[indices]), tol=tol, method=method, jac=grad(obj) if use_grad else None)

                # set beta value to alpha
                if tie_alpha_beta:
                    result.x[4] = result.x[3]
                params, loss, success = result.x, result.fun, result.success

            results_dict[tuple(init_params)] = {'params': params, 'loss': loss}
            param_list.append(params)

            # update best params so far
            if success and loss < best_loss:
                best_loss = loss
                best_params = params

            # add all best 100 results to priority queue
            if len(pq) < 100:
                heapq.heappush(pq, PQItem(loss, params))
            elif loss < pq[0].loss:
                heapq.heappushpop(pq, PQItem(loss, params))

            i += 1
            if i == max_opt_inits:
                break

        largest = heapq.nlargest(100, pq)

        if best_params is not None:
            best_params_untransformed = list(untransform_params(best_params))
            A, B, E, alpha, beta = best_params_untransformed
            print(f"Best fit parameters: A={A}, B={B}, E={E}, alpha={alpha}, beta={beta}")
            print(f"Best loss: {best_loss}")

            param_list = np.array(param_list)
            cov_matrix = np.cov(np.transpose(param_list))
            param_list_untransformed = untransform_params(param_list)
            cov_matrix_untransformed = np.cov(np.transpose(param_list_untransformed))
            standard_errors = np.sqrt(np.diag(cov_matrix[:5, :5]))
            standard_errors_untransformed = np.sqrt(np.diag(cov_matrix_untransformed[:5, :5]))

            parameter_labels = ["A", "B", "E", "alpha", "beta"]
            print("Parameter estimates and their standard errors")
            for index, label in enumerate(parameter_labels):
                print("%s: %.5f (%.5f)" % (label, best_params_untransformed[index], standard_errors_untransformed[index]))

        else:
            print("Optimization failed to converge.")

        return {'pq': pq, 'loss': best_loss, 'params':best_params, 'form': form}
