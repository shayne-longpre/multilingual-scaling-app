import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    Curve,
    CurvePoint,
    CustomFittedParams,
    CustomScalingLawConfig,
    DataPoint,
    FitResponse,
    FittedParams,
    GenerateFormExpPartsRequest,
    GenerateFormExpPartsResponse,
    ScalingLawInfo,
    ValidateCustomLawRequest,
    ValidateCustomLawResponse,
)
from api.utils import auto_generate_form_exp_parts, validate_form_exp_parts
from src.scaling_law_classes.basic_scaling_law import BasicScalingLaw
from src.scaling_law_classes.data_constrained_scaling_law import (
    DataConstrainedScalingLaw,
)
from src.scaling_law_classes.general_scaling_law import GeneralScalingLaw
from src.scaling_laws import ALL_SCALING_LAWS

app = FastAPI(title="Scaling Law Fitting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/scaling-laws")
def get_scaling_laws() -> dict[str, list[ScalingLawInfo]]:
    laws = []
    for name, wrapper in ALL_SCALING_LAWS.items():
        required_columns = ["N", "D", "Loss"]
        if wrapper.extra_args:
            required_columns.extend(wrapper.extra_args)
        laws.append(
            ScalingLawInfo(
                name=name,
                paper_url=wrapper.paper,
                required_columns=required_columns,
                description=wrapper.notes,
            )
        )
    return {"scaling_laws": laws}


@app.post("/api/generate-form-exp-parts")
def generate_form_exp_parts(request: GenerateFormExpPartsRequest) -> GenerateFormExpPartsResponse:
    try:
        form_exp_parts = auto_generate_form_exp_parts(request.form_str, request.params_str)
        return GenerateFormExpPartsResponse(success=True, form_exp_parts=form_exp_parts)
    except ValueError as e:
        return GenerateFormExpPartsResponse(success=False, error=str(e))
    except Exception as e:
        return GenerateFormExpPartsResponse(success=False, error=f"Unexpected error: {e}")


@app.post("/api/validate-custom-law")
def validate_custom_law(request: ValidateCustomLawRequest) -> ValidateCustomLawResponse:
    config = request.config

    # Validate params_str format
    param_names = config.params_str.split()
    if not param_names:
        return ValidateCustomLawResponse(valid=False, error="Parameters string cannot be empty")

    # Validate vars_str format
    var_names = config.vars_str.split()
    if not var_names:
        return ValidateCustomLawResponse(valid=False, error="Variables string cannot be empty")

    # Check for N and D in variables (required for fitting)
    if "N" not in var_names or "D" not in var_names:
        return ValidateCustomLawResponse(
            valid=False,
            error="Variables must include 'N' (model size) and 'D' (training tokens)"
        )

    # Validate form_exp_parts_str
    if not config.form_exp_parts_str:
        return ValidateCustomLawResponse(
            valid=False, error="Log-space expressions are required"
        )

    valid, error = validate_form_exp_parts(
        config.form_str, config.params_str, config.vars_str, config.form_exp_parts_str
    )
    if not valid:
        return ValidateCustomLawResponse(valid=False, error=error)

    # Try to instantiate the GeneralScalingLaw to catch any parsing errors
    try:
        dummy_params = {p: 1.0 for p in param_names}
        GeneralScalingLaw(
            params=dummy_params,
            form_str=config.form_str,
            params_str=config.params_str,
            vars_str=config.vars_str,
            form_exp_parts_str=config.form_exp_parts_str,
        )
    except Exception as e:
        return ValidateCustomLawResponse(valid=False, error=f"Invalid scaling law configuration: {e}")

    return ValidateCustomLawResponse(
        valid=True,
        param_names=param_names,
        var_names=var_names,
    )


@app.post("/api/fit")
async def fit_scaling_law(
    file: UploadFile = File(...),
    scaling_law_name: str = Form(...),
    custom_config: str | None = Form(None),
) -> FitResponse:
    # Handle custom scaling law
    is_custom = scaling_law_name == "Custom"

    if is_custom:
        if not custom_config:
            return FitResponse(
                success=False,
                error="Custom scaling law requires configuration",
            )
        try:
            config = CustomScalingLawConfig.model_validate_json(custom_config)
        except Exception as e:
            return FitResponse(success=False, error=f"Invalid custom config: {e}")

        required_columns = ["Loss"] + config.vars_str.split()
    else:
        if scaling_law_name not in ALL_SCALING_LAWS:
            return FitResponse(
                success=False,
                error=f"Unknown scaling law: {scaling_law_name}. Available: {list(ALL_SCALING_LAWS.keys())}",
            )
        wrapper = ALL_SCALING_LAWS[scaling_law_name]
        required_columns = ["N", "D", "Loss"]
        if wrapper.extra_args:
            required_columns.extend(wrapper.extra_args)

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return FitResponse(success=False, error=f"Failed to parse CSV: {e}")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return FitResponse(
            success=False,
            error=f"Missing required columns: {missing_cols}. Found: {list(df.columns)}",
        )

    if is_custom:
        return await _fit_custom_scaling_law(df, config)
    else:
        return await _fit_predefined_scaling_law(df, wrapper)


async def _fit_custom_scaling_law(df: pd.DataFrame, config: CustomScalingLawConfig) -> FitResponse:
    param_names = config.params_str.split()
    var_names = config.vars_str.split()

    # Create dummy params for instantiation (will be overwritten by fit)
    dummy_params = {p: 1.0 for p in param_names}

    try:
        custom_law = GeneralScalingLaw(
            params=dummy_params,
            form_str=config.form_str,
            params_str=config.params_str,
            vars_str=config.vars_str,
            form_exp_parts_str=config.form_exp_parts_str,
        )
    except Exception as e:
        return FitResponse(success=False, error=f"Failed to create scaling law: {e}")

    try:
        fit_loss, fit_params = custom_law.fit(df, init_params=config.init_params)
    except Exception as e:
        return FitResponse(success=False, error=f"Fitting failed: {e}")

    # Create fitted law with the optimized parameters
    fitted_law = GeneralScalingLaw(
        params=fit_params,
        form_str=config.form_str,
        params_str=config.params_str,
        vars_str=config.vars_str,
        form_exp_parts_str=config.form_exp_parts_str,
        is_fit_already=True,
    )

    # Build original data with predictions
    original_data = []
    for _, row in df.iterrows():
        n_val = float(row["N"])
        d_val = float(row["D"])
        loss_val = float(row["Loss"])
        c_val = 6.0 * n_val * d_val

        # Build kwargs for any extra variables
        extra_kwargs = {v: float(row[v]) for v in var_names if v not in ("N", "D")}
        pred_loss = fitted_law.loss_expr(N=n_val, D=d_val, **extra_kwargs)

        original_data.append(
            DataPoint(
                N=n_val,
                D=d_val,
                C=c_val,
                Loss=loss_val,
                predicted_loss=float(pred_loss),
            )
        )

    # Generate curves
    median_n = float(np.median(df["N"]))
    median_d = float(np.median(df["D"]))
    min_n, max_n = float(df["N"].min()), float(df["N"].max())
    min_d, max_d = float(df["D"].min()), float(df["D"].max())

    # Build extra kwargs for curve generation (use median for extra vars)
    extra_kwargs = {}
    for v in var_names:
        if v not in ("N", "D") and v in df.columns:
            extra_kwargs[v] = float(np.median(df[v]))

    curves_by_n = []
    n_range = np.geomspace(min_n * 0.5, max_n * 2, 100)
    curve_points = [
        CurvePoint(x=float(n), y=float(fitted_law.loss_expr(N=n, D=median_d, **extra_kwargs)))
        for n in n_range
    ]
    curves_by_n.append(Curve(label=f"D = {median_d:.2e}", points=curve_points))

    curves_by_d = []
    d_range = np.geomspace(min_d * 0.5, max_d * 2, 100)
    curve_points = [
        CurvePoint(x=float(d), y=float(fitted_law.loss_expr(N=median_n, D=d, **extra_kwargs)))
        for d in d_range
    ]
    curves_by_d.append(Curve(label=f"N = {median_n:.2e}", points=curve_points))

    # Build formula string with fitted values
    formula_parts = []
    for name, value in sorted(fit_params.items()):
        formula_parts.append(f"{name} = {value:.4g}")
    formula = f"{config.form_str} where " + ", ".join(formula_parts)

    custom_fitted_params = CustomFittedParams(
        params={k: float(v) for k, v in fit_params.items()},
        formula=formula,
    )

    return FitResponse(
        success=True,
        fit_loss=float(fit_loss),
        custom_fitted_params=custom_fitted_params,
        original_data=original_data,
        curves_by_N=curves_by_n,
        curves_by_D=curves_by_d,
        curves_by_C=[],  # Custom laws don't support optimal allocation yet
        formula=formula,
    )


async def _fit_predefined_scaling_law(df: pd.DataFrame, wrapper) -> FitResponse:
    try:
        if wrapper.use_init_params:
            init_params = wrapper.scaling_law.params
            fit_loss, fit_params = wrapper.scaling_law.fit(df, init_params=init_params)
        else:
            fit_loss, fit_params = wrapper.scaling_law.fit(df)
    except Exception as e:
        return FitResponse(success=False, error=f"Fitting failed: {e}")

    fitted_params = FittedParams(
        A=float(fit_params["A"]),
        B=float(fit_params["B"]),
        irreducible=float(fit_params["E"]),
        alpha=float(fit_params["alpha"]),
        beta=float(fit_params["beta"]),
        rd_star=float(fit_params["rd_star"]) if "rd_star" in fit_params else None,
        rn_star=float(fit_params["rn_star"]) if "rn_star" in fit_params else None,
    )

    if isinstance(wrapper.scaling_law, DataConstrainedScalingLaw):
        fitted_law = DataConstrainedScalingLaw(params=fit_params)
    else:
        fitted_law = BasicScalingLaw(params=fit_params)

    original_data = []
    for _, row in df.iterrows():
        n_val = float(row["N"])
        d_val = float(row["D"])
        loss_val = float(row["Loss"])
        c_val = 6.0 * n_val * d_val

        if isinstance(fitted_law, DataConstrainedScalingLaw):
            u_val = float(row["U"])
            pred_loss = fitted_law.loss(N=n_val, D=d_val, U=u_val)
        else:
            pred_loss = fitted_law.loss(N=n_val, D=d_val)

        original_data.append(
            DataPoint(
                N=n_val,
                D=d_val,
                C=c_val,
                Loss=loss_val,
                predicted_loss=float(pred_loss),
            )
        )

    median_n = float(np.median(df["N"]))
    median_d = float(np.median(df["D"]))
    min_n, max_n = float(df["N"].min()), float(df["N"].max())
    min_d, max_d = float(df["D"].min()), float(df["D"].max())

    curves_by_n = []
    n_range = np.geomspace(min_n * 0.5, max_n * 2, 100)
    if isinstance(fitted_law, DataConstrainedScalingLaw):
        median_u = float(np.median(df["U"]))
        curve_points = [
            CurvePoint(x=float(n), y=float(fitted_law.loss(N=n, D=median_d, U=median_u)))
            for n in n_range
        ]
    else:
        curve_points = [
            CurvePoint(x=float(n), y=float(fitted_law.loss(N=n, D=median_d)))
            for n in n_range
        ]
    curves_by_n.append(Curve(label=f"D = {median_d:.2e}", points=curve_points))

    curves_by_d = []
    d_range = np.geomspace(min_d * 0.5, max_d * 2, 100)
    if isinstance(fitted_law, DataConstrainedScalingLaw):
        curve_points = [
            CurvePoint(x=float(d), y=float(fitted_law.loss(N=median_n, D=d, U=median_u)))
            for d in d_range
        ]
    else:
        curve_points = [
            CurvePoint(x=float(d), y=float(fitted_law.loss(N=median_n, D=d)))
            for d in d_range
        ]
    curves_by_d.append(Curve(label=f"N = {median_n:.2e}", points=curve_points))

    curves_by_c = []
    min_c = 6.0 * min_n * min_d
    max_c = 6.0 * max_n * max_d
    c_range = np.geomspace(min_c * 0.5, max_c * 2, 100)

    curve_points = []
    for c_val in c_range:
        try:
            result = fitted_law.compute_optimal_allocation(c_val)
            curve_points.append(CurvePoint(x=float(c_val), y=float(result["loss"])))
        except Exception:
            continue
    if curve_points:
        curves_by_c.append(Curve(label="Optimal allocation", points=curve_points))

    formula = f"L = {fitted_params.irreducible:.4f} + {fitted_params.A:.2f}/N^{fitted_params.alpha:.4f} + {fitted_params.B:.2f}/D^{fitted_params.beta:.4f}"
    if fitted_params.rd_star is not None:
        formula += f" (rd* = {fitted_params.rd_star:.4f}, rn* = {fitted_params.rn_star:.4f})"

    return FitResponse(
        success=True,
        fit_loss=float(fit_loss),
        fitted_params=fitted_params,
        original_data=original_data,
        curves_by_N=curves_by_n,
        curves_by_D=curves_by_d,
        curves_by_C=curves_by_c,
        formula=formula,
    )
