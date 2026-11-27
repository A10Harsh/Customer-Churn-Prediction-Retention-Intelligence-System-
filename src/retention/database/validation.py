"""Validate the 19 frozen predictor fields without a target or identifier."""

from decimal import Decimal, InvalidOperation

from retention.data.schemas import INTERNET_OPTIONS, PAYMENT_METHODS, YES_NO

CATEGORIES = {
    "gender": ("Male", "Female"),
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    **{
        name: INTERNET_OPTIONS
        for name in (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        )
    },
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": YES_NO,
    "PaymentMethod": PAYMENT_METHODS,
}
NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
COLUMNS = tuple(CATEGORIES) + NUMERIC
COLUMN_MAP = {
    "SeniorCitizen": "senior_citizen",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
}


def validate_predictors(values: dict) -> dict:
    if set(values) != set(COLUMNS):
        raise ValueError("Expected exactly the 19 predictor fields")
    result = {}
    for key, allowed in CATEGORIES.items():
        if values[key] not in allowed:
            raise ValueError(f"Invalid category: {key}")
        result[key] = values[key]
    for key in NUMERIC:
        if isinstance(values[key], bool):
            raise ValueError(f"Invalid numeric predictor: {key}")
        try:
            number = Decimal(str(values[key]))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"Invalid numeric predictor: {key}") from exc
        if not number.is_finite() or number < 0:
            raise ValueError(f"Invalid numeric predictor: {key}")
        if key in ("SeniorCitizen", "tenure"):
            if number != number.to_integral_value() or (
                key == "SeniorCitizen" and number not in (0, 1)
            ):
                raise ValueError(f"Invalid integer predictor: {key}")
            result[key] = int(number)
        else:
            if number.as_tuple().exponent < -4 or number >= (
                10**8 if key == "MonthlyCharges" else 10**10
            ):
                raise ValueError(f"Charge exceeds database precision: {key}")
            result[key] = number
    if (result["PhoneService"] == "No") != (
        result["MultipleLines"] == "No phone service"
    ):
        raise ValueError("Phone service and multiple lines disagree")
    for key in (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ):
        if (result["InternetService"] == "No") != (
            result[key] == "No internet service"
        ):
            raise ValueError(f"Internet service and {key} disagree")
    return result


def model_values(predictors: dict) -> dict:
    return {
        COLUMN_MAP.get(key, key.lower() if key[0].isupper() else key): value
        for key, value in predictors.items()
    }


def customer_predictors(customer) -> dict:
    return {
        key: getattr(
            customer, COLUMN_MAP.get(key, key.lower() if key[0].isupper() else key)
        )
        for key in COLUMNS
    }
