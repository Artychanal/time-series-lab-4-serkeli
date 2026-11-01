from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from clustering import category_similarity, cluster_analysis, windows
from correlation_analysis import correlation_tables, lagged_associations, leave_one_out_correlations
from data_loader import load_sales, load_steam
from decomposition import decompose, sales_trend, steam_sensitivity
from evaluation import compare_ensembles, describe, effectiveness, fixed_cluster_properties, forecast_comparison, sales_properties, stationarity, steam_properties
from synthetic_data import fit_steam, generate_sales, generate_steam
from visualization import architecture, create_figures


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_analysis(data_path: Path, output: Path, simulations=300, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    log = []
    def progress(message):
        print(message, flush=True)
        log.append(message)
    def save(frame, name, index=False):
        frame.to_csv(output / f"{name}.csv", index=index)
    progress("Завантаження та перевірка двох наборів...")
    steam, steam_quality = load_steam(data_path / "cyberpunk_2077_monthly_reviews.csv")
    sales, daily, sales_quality = load_sales(data_path / "dataset8.csv")
    save(steam, "steam_clean")
    save(sales, "sales_clean")
    save(daily, "sales_daily_observed", True)
    calendar = daily.reindex(pd.date_range("2015-01-01", "2015-01-31", freq="D"))
    calendar.index.name = "date"
    calendar["observed"] = calendar.index.isin(daily.index)
    save(calendar, "sales_calendar_unknown", True)
    zero_scenario = calendar.copy()
    zero_scenario[daily.columns] = zero_scenario[daily.columns].fillna(0)
    save(zero_scenario, "sales_calendar_zero_assumption", True)
    descriptive = []
    for dataset, frame, cols in [("steam", steam, ["total_reviews", "positive_reviews", "negative_reviews", "positive_share"]),
                                  ("sales", sales, ["price", "quantity", "revenue", "shipping", "shipping_rate"])]:
        for col in cols:
            descriptive.append({"dataset": dataset, "variable": col, **describe(frame[col])})
    save(pd.DataFrame(descriptive), "descriptive_statistics")
    progress("Декомпозиція, властивості та кореляції...")
    dec, dec_info = decompose(np.log1p(steam.total_reviews), steam.date)
    save(dec, "steam_decomposition")
    sensitivity = steam_sensitivity(steam)
    save(sensitivity, "steam_decomposition_sensitivity")
    stat = pd.DataFrame([stationarity(np.log1p(steam.total_reviews), "log_total"),
                         stationarity(np.diff(np.log1p(steam.total_reviews)), "diff_log_total"),
                         stationarity(dec.residual, "stl_residual")])
    save(stat, "stationarity")
    sales_dec, sales_loo, sales_dec_info = sales_trend(daily)
    save(sales_dec, "sales_decomposition")
    save(sales_loo, "sales_trend_leave_date_out")
    sales_dec_info.update({"loo_slope_min": float(sales_loo.slope.min()), "loo_slope_max": float(sales_loo.slope.max()),
                          "loo_linear_rmse": float(np.sqrt(np.mean(sales_loo.linear_error**2))),
                          "loo_constant_rmse": float(np.sqrt(np.mean(sales_loo.constant_error**2)))})
    steam_numeric = steam.drop(columns="date")
    steam_corr = correlation_tables(steam_numeric, "steam_levels", output)
    diff_corr = correlation_tables(steam_numeric.diff().dropna(), "steam_differences", output)
    log_cols = np.log1p(steam[["positive_reviews", "negative_reviews", "total_reviews"]])
    residuals = pd.DataFrame({c: decompose(log_cols[c], steam.date)[0].residual for c in log_cols})
    residuals["positive_share"] = decompose(steam.positive_share, steam.date)[0].residual
    correlation_tables(residuals, "steam_stl_residuals", output)
    sales_numeric = sales[["price", "quantity", "revenue", "shipping", "shipping_rate"]]
    sales_corr = correlation_tables(sales_numeric, "sales_operations", output)
    correlation_tables(daily, "sales_observed_dates", output)
    save(lagged_associations(steam), "steam_lagged_changes")
    corr_loo = leave_one_out_correlations(sales, [("price", "quantity"), ("price", "shipping_rate"), ("revenue", "shipping")])
    save(corr_loo, "sales_correlation_sensitivity")
    progress("Кластеризація та перевірка стійкості...")
    feature_columns = ["level", "volatility", "slope", "share"]
    window_frame, profiles = windows(steam)
    labels, search, stability, cluster_steam = cluster_analysis(window_frame[feature_columns], seed)
    window_frame["cluster"] = labels
    save(window_frame, "steam_windows")
    save(pd.DataFrame(profiles), "steam_window_profiles")
    save(search, "steam_cluster_search")
    save(stability, "steam_cluster_stability")
    save(window_frame.groupby("cluster")[feature_columns].mean(), "steam_cluster_centers", True)
    shape_labels, shape_search, shape_stability, shape_info = cluster_analysis(profiles, seed)
    save(shape_search, "steam_shape_cluster_search")
    save(shape_stability, "steam_shape_cluster_stability")
    save(pd.DataFrame({"start": window_frame.start, "feature_cluster": labels, "shape_cluster": shape_labels}), "steam_shape_assignments")
    cluster_steam["feature_shape_ari"] = float(adjusted_rand_score(labels, shape_labels))
    step_rows = []
    for step in [1, 6, 12]:
        other, _ = windows(steam, step=step)
        other_labels, _, _, info = cluster_analysis(other[feature_columns], seed, max_k=3)
        common = window_frame[["start", "cluster"]].merge(other[["start"]].assign(other_cluster=other_labels), on="start")
        step_rows.append({"step": step, **info, "common_windows": len(common),
                          "common_ari": adjusted_rand_score(common.cluster, common.other_cluster) if len(common) > 1 else np.nan})
    save(pd.DataFrame(step_rows), "steam_window_step_sensitivity")
    sale_features = sales[["price", "quantity", "shipping_rate"]]
    sales_labels, search_sales, stability_sales, cluster_sales = cluster_analysis(sale_features, seed, max_k=3)
    save(search_sales, "sales_cluster_search")
    save(stability_sales, "sales_cluster_stability")
    labeled_sales = sales.assign(cluster=sales_labels)
    save(labeled_sales, "sales_cluster_assignments")
    save(labeled_sales.groupby("cluster")[["price", "quantity", "shipping_rate", "revenue"]].agg(["mean", "size"]), "sales_cluster_centers", True)
    category_profiles, category_distance = category_similarity(sales)
    save(category_profiles, "sales_category_profiles_observed_dates", True)
    save(category_distance, "sales_category_profile_distances", True)
    progress("Вибір часової моделі на валідації, окремий фінальний тест...")
    predictions, comparison, chosen = forecast_comparison(steam)
    save(predictions, "steam_chronological_predictions")
    save(comparison, "steam_model_comparison")
    model = fit_steam(steam, chosen)
    progress(f"Обрана модель: {chosen}. Генерування {simulations} реалізацій кожного варіанта...")
    steam_real, sales_real = steam_properties(steam), sales_properties(sales, daily.index)
    steam_reference = window_frame[feature_columns].to_numpy(float)
    sales_reference = sale_features.to_numpy(float)
    steam_real.update(fixed_cluster_properties(steam_reference, steam_reference, labels))
    sales_real.update(fixed_cluster_properties(sales_reference, sales_reference, sales_labels))
    records_steam, records_sales, examples = [], [], {}
    max_steam_identity_error, max_sales_identity_error = 0, 0.0
    for generator_index, generator in enumerate(["structured", "independent"]):
        for replication in range(simulations):
            steam_rng = np.random.default_rng(np.random.SeedSequence([seed, generator_index, replication, 1]))
            sales_rng = np.random.default_rng(np.random.SeedSequence([seed, generator_index, replication, 2]))
            generated_steam = generate_steam(model, steam.date, steam_rng, baseline=generator == "independent")
            generated_sales = generate_sales(sales, sales_rng, baseline=generator == "independent")
            generated_windows, _ = windows(generated_steam)
            steam_metrics = {**steam_properties(generated_steam),
                             **fixed_cluster_properties(generated_windows[feature_columns], steam_reference, labels)}
            sales_metrics = {**sales_properties(generated_sales, daily.index),
                             **fixed_cluster_properties(generated_sales[["price", "quantity", "shipping_rate"]], sales_reference, sales_labels)}
            records_steam.append({"generator": generator, "replication": replication, **steam_metrics})
            records_sales.append({"generator": generator, "replication": replication, **sales_metrics})
            max_steam_identity_error = max(max_steam_identity_error, int(np.abs(generated_steam.total_reviews - generated_steam.positive_reviews - generated_steam.negative_reviews).max()))
            max_sales_identity_error = max(max_sales_identity_error, float(np.abs(generated_sales.revenue - generated_sales.price * generated_sales.quantity).max()))
            if replication == 0:
                examples[f"steam_{generator}"] = generated_steam
                examples[f"sales_{generator}"] = generated_sales
                save(generated_steam, f"synthetic_steam_{generator}")
                save(generated_sales, f"synthetic_sales_{generator}")
        progress(f"Завершено: {generator}, {simulations} реалізацій для кожного набору")
    ensembles = {}
    for name, real, records in [("steam", steam_real, records_steam), ("sales", sales_real, records_sales)]:
        individual, summary = compare_ensembles(real, records)
        save(individual, f"{name}_ensemble_individual")
        save(summary, f"{name}_ensemble_comparison")
        ensembles[name] = summary
    effectiveness_tables = {
        "steam": effectiveness(records_steam, steam_real, ["acf1", "corr_activity_share", "cluster_mean_distance"]),
        "sales": effectiveness(records_sales, sales_real, ["corr_price_quantity", "corr_price_shipping_rate", "cluster_mean_distance"]),
    }
    for name, table in effectiveness_tables.items():
        save(table, f"{name}_generator_effectiveness")
    applicability = pd.DataFrame([
        {"dataset": "Steam", "method": "STL period=12", "status": "exploratory", "reason": "69 regular months; fewer than six cycles"},
        {"dataset": "Steam", "method": "Window clustering", "status": "descriptive", "reason": "20 overlapping windows, not independent"},
        {"dataset": "DataSet8", "method": "Seasonal STL / ADF / KPSS", "status": "not reliable", "reason": "9 irregular dates in one month"},
        {"dataset": "DataSet8", "method": "Linear plus residual", "status": "descriptive", "reason": "Calendar-time regression on observed dates only"},
        {"dataset": "DataSet8", "method": "Operation clustering", "status": "exploratory", "reason": "15 rows and 3 selected features"},
        {"dataset": "DataSet8", "method": "Category temporal profiles", "status": "conditional similarity", "reason": "Sparse profiles on nine observed dates; not full calendar"},
    ])
    save(applicability, "method_applicability")
    checks = {"steam_rows": len(steam), "sales_rows": len(sales), "sales_dates": len(daily),
              "stl_reconstruction_error": dec_info["reconstruction_error"],
              "synthetic_steam_identity_max_error": max_steam_identity_error,
              "synthetic_sales_identity_max_error": max_sales_identity_error,
              "validation_last_index": int(predictions.loc[predictions.phase == "validation", "index"].max()),
              "test_first_index": int(predictions.loc[predictions.phase == "test", "index"].min()),
              "nonfinite_simulation_metrics": int(pd.DataFrame(records_steam).select_dtypes("number").isna().sum().sum() + pd.DataFrame(records_sales).select_dtypes("number").isna().sum().sum())}
    if checks["stl_reconstruction_error"] > 1e-9 or max_steam_identity_error or max_sales_identity_error > .011 or checks["nonfinite_simulation_metrics"]:
        raise AssertionError(f"Verification failed: {checks}")
    progress("Збереження графіків і підсумків...")
    create_figures(steam, sales, daily, dec, sales_dec, predictions, window_frame, profiles, sales_labels,
                   {"steam": steam_corr["pearson"], "differences": diff_corr["spearman"], "sales": sales_corr["spearman"]},
                   category_distance, ensembles, examples, output / "figures")
    architecture(output / "figures/15_architecture.png")
    summary = {"seed": seed, "simulations_per_generator_per_dataset": simulations,
               "data_quality": {"steam": steam_quality, "sales": sales_quality},
               "decomposition": {"steam": dec_info, "sales": sales_dec_info},
               "stationarity": stat.to_dict("records"), "descriptive": descriptive,
               "clusters": {"steam": cluster_steam, "steam_shape": shape_info, "sales": cluster_sales},
               "selected_generator_model": chosen, "generator_parameters": {"beta": model.beta, "phi": model.phi},
               "model_comparison": comparison.to_dict("records"), "verification": checks,
               "ensemble_comparisons": {k: v.to_dict("records") for k, v in ensembles.items()},
               "generator_effectiveness": {k: v.to_dict("records") for k, v in effectiveness_tables.items()},
               "protocol": {"initial_train": 36, "validation_months": len(steam) - 48, "test_months": 12,
                            "selection": "Expanding-window validation; choose among three parametric generators before test",
                            "test": "Fit once before test; score one-step predictions using only previous observations",
                            "synthetic": "Conditional full-history fit; ensemble percentile ranges are not confidence intervals",
                            "sales": "Smoothed empirical transactions, dates sampled on observed support; no claims beyond January"},
               "limitations": ["Only 69 Steam months; fewer than six annual cycles", "Overlapping windows are dependent",
                               "DataSet8 has 15 operations on nine irregular dates", "Unknown sales dates are not zero by default",
                               "Sales synthetic fit is conditional and cannot establish out-of-sample adequacy",
                               "STL components and cluster separation do not establish causal mechanisms",
                               "Synthetic percentile intervals condition on fitted parameters; parameter uncertainty omitted",
                               "Sales generation preserves observed catalog prices and empirical product frequencies; dates independently resampled"],
               "runtime": {"python": platform.python_version(), **{x: version(x) for x in ["numpy", "pandas", "scipy", "matplotlib", "statsmodels", "scikit-learn"]}}}
    (output / "analysis_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    progress(comparison.round(5).to_string(index=False))
    progress(f"Готово: {output.resolve()}")
    (output / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    return summary
