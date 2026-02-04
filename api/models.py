from pydantic import BaseModel, ConfigDict, Field


class CustomScalingLawConfig(BaseModel):
    form_str: str = Field(description="Loss formula, e.g., 'E + A / N**alpha + B / D**beta'")
    params_str: str = Field(description="Space-separated parameter names, e.g., 'A B E alpha beta'")
    vars_str: str = Field(default="N D", description="Space-separated variable names, e.g., 'N D'")
    form_exp_parts_str: list[str] = Field(
        description="Log-space expressions for each additive term, e.g., ['logA - alpha * logN', 'logB - beta * logD', 'logE']"
    )
    init_params: dict[str, float] | None = Field(
        default=None, description="Optional initial parameter values for optimization"
    )

    model_config = ConfigDict(populate_by_name=True)


class CustomFittedParams(BaseModel):
    params: dict[str, float] = Field(description="Fitted parameter name to value mapping")
    formula: str = Field(description="Rendered formula string with fitted values")

    model_config = ConfigDict(populate_by_name=True)


class ScalingLawInfo(BaseModel):
    name: str
    paper_url: str
    required_columns: list[str]
    description: str

    model_config = ConfigDict(populate_by_name=True)


class FittedParams(BaseModel):
    A: float
    B: float
    irreducible: float
    alpha: float
    beta: float
    rd_star: float | None = None
    rn_star: float | None = None

    model_config = ConfigDict(populate_by_name=True)


class DataPoint(BaseModel):
    N: float
    D: float
    C: float
    Loss: float
    predicted_loss: float

    model_config = ConfigDict(populate_by_name=True)


class CurvePoint(BaseModel):
    x: float
    y: float

    model_config = ConfigDict(populate_by_name=True)


class Curve(BaseModel):
    label: str
    points: list[CurvePoint]

    model_config = ConfigDict(populate_by_name=True)


class FitResponse(BaseModel):
    success: bool
    error: str | None = None
    fit_loss: float | None = None
    fitted_params: FittedParams | None = None
    custom_fitted_params: CustomFittedParams | None = None
    original_data: list[DataPoint] | None = None
    curves_by_N: list[Curve] | None = None
    curves_by_D: list[Curve] | None = None
    curves_by_C: list[Curve] | None = None
    formula: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class GenerateFormExpPartsRequest(BaseModel):
    form_str: str
    params_str: str

    model_config = ConfigDict(populate_by_name=True)


class GenerateFormExpPartsResponse(BaseModel):
    success: bool
    form_exp_parts: list[str] | None = None
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class ValidateCustomLawRequest(BaseModel):
    config: CustomScalingLawConfig

    model_config = ConfigDict(populate_by_name=True)


class ValidateCustomLawResponse(BaseModel):
    valid: bool
    error: str | None = None
    param_names: list[str] | None = None
    var_names: list[str] | None = None

    model_config = ConfigDict(populate_by_name=True)
