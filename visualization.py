from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 170})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def heatmap(matrix, title, path, limits=(-1, 1)):
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=limits[0], vmax=limits[1])
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix.iloc[i, j] > .7 * limits[1] or matrix.iloc[i, j] < -.7 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, path)


def ensemble_plot(table, metrics, title, path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        group = table.loc[table.metric == metric]
        for i, row in enumerate(group.itertuples()):
            ax.plot([i, i], [row.synthetic_q025, row.synthetic_q975], color="C0", lw=4, alpha=.6)
            ax.scatter(i, row.synthetic_mean, color="C0", zorder=3)
        ax.axhline(group.real.iloc[0], color="C3", ls="--", label="Реальні дані")
        ax.set_xticks(range(len(group)), group.generator, rotation=15)
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    save(fig, path)


def create_figures(steam, sales, daily, decomposition, sales_dec, predictions, window_table,
                   profiles, sales_labels, matrices, category_distance, ensembles, examples, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steam.date, steam.total_reviews, marker=".")
    axes[0].set_ylabel("Відгуки")
    axes[0].set_title("Cyberpunk 2077: збережено всі реальні сплески")
    axes[1].plot(steam.date, np.log1p(steam.total_reviews), label="log1p(total)")
    axes[1].plot(steam.date, np.log1p(steam.total_reviews).rolling(12, min_periods=6).mean(), label="Ковзне середнє 12 міс.")
    axes[1].legend()
    axes[2].plot(steam.date, steam.positive_share, color="C2")
    axes[2].set_ylabel("Частка позитивних")
    save(fig, output / "01_steam_series.png")
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for ax, column, title in zip(axes, ["observed", "trend", "seasonal", "residual"],
                                 ["log1p кількості", "Тренд", "Сезонна компонента, period=12", "Залишок"]):
        ax.plot(decomposition.date, decomposition[column])
        ax.set_ylabel(title)
    axes[1].plot(decomposition.date, decomposition.rolling_trend, ls="--", label="Ковзна медіана 13 міс.")
    axes[1].legend()
    save(fig, output / "02_steam_decomposition.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    y = np.log1p(steam.total_reviews)
    plot_acf(y, lags=18, ax=axes[0, 0], zero=False, title="ACF log1p")
    plot_pacf(y, lags=12, ax=axes[0, 1], zero=False, method="ywm", title="PACF log1p")
    plot_acf(np.diff(y), lags=18, ax=axes[1, 0], zero=False, title="ACF перших різниць")
    plot_acf(decomposition.residual, lags=18, ax=axes[1, 1], zero=False, title="ACF залишку STL")
    save(fig, output / "03_steam_acf.png")
    heatmap(matrices["steam"], "Steam: Pearson у рівнях", output / "04_steam_correlations.png")
    heatmap(matrices["differences"], "Steam: Spearman перших різниць", output / "05_steam_changes.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for group in sorted(window_table.cluster.unique()):
        keep = window_table.cluster.to_numpy() == group
        for profile in profiles[keep]:
            axes[0].plot(np.arange(1, 13), profile, alpha=.25, color=f"C{group}")
        axes[0].plot(np.arange(1, 13), profiles[keep].mean(axis=0), lw=2, color=f"C{group}", label=f"Група {group + 1}")
    axes[0].set(xlabel="Місяць у вікні", ylabel="Нормований log1p", title="Профілі за багатофакторними кластерами")
    axes[0].legend()
    axes[1].scatter(window_table.level, window_table.volatility, c=[f"C{g}" for g in window_table.cluster], s=65)
    axes[1].set(xlabel="Середній log1p", ylabel="Стандартне відхилення log1p", title="12-місячні вікна, крок 3 місяці")
    save(fig, output / "06_steam_clusters.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    selected = predictions[predictions.phase == "test"]
    for name, group in selected.groupby("model"):
        ax.plot(group.date, group.predicted_log, marker=".", label=name)
    real = selected[selected.model == "naive"]
    ax.plot(real.date, real.actual_log, color="black", lw=2, label="Факт")
    ax.set(ylabel="log1p відгуків", title="Окремі фінальні 12 місяців: однокрокові прогнози")
    ax.legend(ncol=3, fontsize=8)
    save(fig, output / "07_steam_validation.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].bar(daily.index, daily.revenue, width=.65)
    axes[0].scatter(sales_dec.date, sales_dec.linear_component, color="C3", label="Лінійна компонента на наявних датах")
    axes[0].set(ylabel="Виручка, USD", title="DataSet 8: 9 дат із записами; інші дати невідомі")
    axes[0].legend(fontsize=8)
    revenue = sales.groupby("category").revenue.sum().sort_values()
    axes[1].barh(revenue.index, revenue.values)
    axes[1].set_xlabel("Виручка категорії, USD")
    save(fig, output / "08_sales_series.png")
    heatmap(matrices["sales"], "DataSet 8: Spearman, 15 операцій", output / "09_sales_correlations.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(sales.price, sales.quantity, c=[f"C{g}" for g in sales_labels], s=70)
    axes[0].set(xlabel="Ціна, USD", ylabel="Кількість", title="Багатофакторні групи операцій")
    axes[1].scatter(sales.price, sales.shipping_rate, c=[f"C{g}" for g in sales_labels], s=70)
    axes[1].set(xlabel="Ціна, USD", ylabel="Доставка / виручка", title="Третя ознака кластеризації")
    save(fig, output / "10_sales_clusters.png")
    heatmap(category_distance, "Відстань нормованих профілів категорій на 9 датах", output / "11_category_similarity.png", (0, 1.42))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(steam.date, np.log1p(steam.total_reviews), color="black", label="Реальні")
    for name in ("structured", "independent"):
        axes[0].plot(steam.date, np.log1p(examples[f"steam_{name}"].total_reviews), alpha=.75, label=name)
    axes[0].set(ylabel="log1p відгуків", title="Незалежні приклади синтетичних реалізацій")
    axes[0].legend()
    for name in ("structured", "independent"):
        frame = examples[f"sales_{name}"]
        axes[1].scatter(frame.price, frame.quantity, label=name, alpha=.65)
    axes[1].scatter(sales.price, sales.quantity, color="black", marker="x", label="Реальні")
    axes[1].set(xlabel="Ціна", ylabel="Кількість", title="Синтетичні операції продажу")
    axes[1].legend()
    save(fig, output / "12_synthetic_examples.png")
    ensemble_plot(ensembles["steam"], ["acf1", "corr_activity_share", "seasonal_strength"],
                  "Steam: 95% інтервали реалізацій, не довірчі інтервали параметра", output / "13_steam_adequacy.png")
    ensemble_plot(ensembles["sales"], ["corr_price_quantity", "corr_price_shipping_rate", "daily_max_revenue_share"],
                  "DataSet 8: умовна відповідність виданій вибірці", output / "14_sales_adequacy.png")


def architecture(path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    nodes = [(0.05, .70, "Steam / DataSet 8\nПеревірка джерел"), (.39, .70, "Декомпозиція\nВластивості та кореляції"),
             (.73, .70, "Подібність\nКластеризація"), (.05, .19, "Вибір моделі\nВалідація за часом"),
             (.39, .19, "Синтетика\n300 реалізацій"), (.73, .19, "Оцінювання\nCSV / JSON / графіки")]
    for x, y, text in nodes:
        ax.text(x + .11, y, text, ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=.7", "fc": "#eef1f4", "ec": "#596875"})
    for x1, y1, x2, y2 in [(.28, .7, .36, .7), (.62, .7, .70, .7), (.84, .57, .16, .33),
                             (.28, .19, .36, .19), (.62, .19, .70, .19)]:
        ax.annotate("", (x2, y2), (x1, y1), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#596875"})
    save(fig, path)
