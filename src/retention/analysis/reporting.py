"""Deterministic JSON, CSV, and HTML report persistence."""

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively convert pandas/NumPy values and non-finite numbers to valid JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return json_safe(
            value.to_dict(orient="index")
            if isinstance(value, pd.DataFrame)
            else value.to_dict()
        )
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or pd.isna(value):
        return None
    return value


def write_summary(summary: dict[str, Any], path: Path) -> None:
    """Write strict, sorted, timestamp-free JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_tables(
    tables: dict[str, pd.DataFrame], directory: Path, *, index: bool = True
) -> list[str]:
    """Write stable CSV representations of analytical tables."""
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for name, table in sorted(tables.items()):
        filename = f"{name}.csv"
        table.to_csv(
            directory / filename,
            index=index,
            lineterminator="\n",
            float_format="%.12g",
        )
        names.append(filename)
    return names


def write_html(
    summary: dict[str, Any], tables: list[str], figures: list[str], path: Path
) -> None:
    """Write a concise standalone executive index with relative artifact links."""
    kpis = summary["business_kpis"]
    finding_rows = "".join(
        f"<li><strong>{escape(str(k))}</strong>: {escape(str(v))}</li>"
        for k, v in kpis.items()
    )
    table_links = "".join(
        f'<li><a href="tables/{escape(x)}">{escape(x)}</a></li>' for x in tables
    )
    images = "".join(
        f'<figure><img src="figures/{escape(x)}" alt="{escape(x)}">'
        f"<figcaption>{escape(x)}</figcaption></figure>"
        for x in figures
    )
    limitations = "".join(f"<li>{escape(x)}</li>" for x in summary["limitations"])
    contract_v = summary["key_findings"]["contract_cramers_v"]
    tenure_pearson = summary["key_findings"]["tenure_total_charges_pearson"]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Automated EDA report</title>
<style>
body{{font:16px system-ui;max-width:1100px;margin:auto;padding:2rem;color:#222}}
img{{max-width:100%}} figure{{margin:2rem 0}}
</style></head><body>
<h1>Customer Retention Intelligence — Automated EDA</h1>
<p>Development population only. Descriptive associations are not causal effects.</p>
<h2>Business KPIs</h2><ul>{finding_rows}</ul>
<h2>Statistical findings</h2>
<p>Contract–churn corrected Cramér's V: {contract_v:.4f}.
Tenure–TotalCharges Pearson correlation: {tenure_pearson:.4f}.</p>
<h2>Figures</h2>{images}
<h2>Machine-readable tables</h2><ul>{table_links}</ul>
<p><a href="summary.json">summary.json</a></p>
<h2>Limitations</h2><ul>{limitations}</ul></body></html>"""
    path.write_text(html, encoding="utf-8")


def write_statistical_html(
    summary: dict[str, Any], tables: list[str], path: Path
) -> None:
    """Write the standalone statistical-analysis report without volatile metadata."""
    findings = summary["key_findings"]
    families = summary["hypothesis_families"]
    family_rows = "".join(
        "<tr>"
        f"<td>{escape(str(name))}</td>"
        f"<td>{int(values['test_count'])}</td>"
        f"<td>{int(values['bh_rejections'])}</td>"
        f"<td>{int(values['holm_rejections'])}</td>"
        "</tr>"
        for name, values in families.items()
    )
    table_links = "".join(
        f'<li><a href="{escape(name)}">{escape(name)}</a></li>' for name in tables
    )
    recommendations = "".join(
        f"<li>{escape(item)}</li>" for item in summary["recommendations"]
    )
    limitations = "".join(f"<li>{escape(item)}</li>" for item in summary["limitations"])
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Automated statistical analysis</title>
<style>
body{{font:16px system-ui;max-width:1050px;margin:auto;padding:2rem;color:#222}}
table{{border-collapse:collapse}}
th,td{{border:1px solid #ccc;padding:.5rem;text-align:left}}
code{{background:#f4f4f4;padding:.1rem .25rem}}
</style></head><body>
<h1>Customer Retention Intelligence — Statistical Analysis</h1>
<p>Development population only. Results are exploratory associations, not causal or
independently confirmatory evidence.</p>
<h2>Statistical evidence and uncertainty</h2>
<p>Overall churn rate: {findings["overall_churn_rate"]:.4f}
({summary["confidence_level"]:.0%} Wilson interval
{findings["overall_churn_ci_lower"]:.4f}–{findings["overall_churn_ci_upper"]:.4f}).
Primary feature–churn decisions changed after BH: {findings["primary_bh_changes"]}.</p>
<table><thead><tr><th>Hypothesis family</th><th>Tests</th><th>BH rejections</th>
<th>Holm rejections</th></tr></thead><tbody>{family_rows}</tbody></table>
<h2>Effect sizes</h2>
<p>Tenure Cohen's d (churned minus retained): {findings["tenure_cohens_d"]:.4f}.
Contract–churn corrected Cramér's V: {findings["contract_cramers_v"]:.4f}.</p>
<h2>Predictor dependencies</h2>
<p>Numerical TotalCharges VIF: {findings["total_charges_vif"]:.4f}.
The temporary mixed design has rank {findings["mixed_design_rank"]} of
{findings["mixed_design_columns_with_intercept"]} columns including the intercept,
with {findings["mixed_design_infinite_vif_columns"]} columns participating in exact
dependencies. Its standardized condition number is explicitly
{escape(str(findings["mixed_design_condition_number_display"]))}.</p>
<h2>Modelling recommendations</h2><ul>{recommendations}</ul>
<h2>Machine-readable artifacts</h2><ul>{table_links}</ul>
<p><a href="summary.json">summary.json</a></p>
<h2>Limitations</h2><ul>{limitations}</ul>
</body></html>"""
    path.write_text(html, encoding="utf-8")
