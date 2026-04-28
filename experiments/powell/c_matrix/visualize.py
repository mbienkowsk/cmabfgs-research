import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    # ensure project root is on the path when running from any cwd
    _root = Path(__file__).parent.parent.parent.parent.resolve()
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import marimo as mo
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import pandas as pd

    from lib.serde import aggregate_convergence_series

    return Path, aggregate_convergence_series, cm, mcolors, mo, pd, plt


@app.cell
def _(Path):
    RESULTS_DIR = Path(__file__).parent / "results"
    return (RESULTS_DIR,)


@app.cell
def _(RESULTS_DIR, pd):
    def rot_dir_to_idx(rot_dir: str):
        return None if rot_dir == "no_rotation" else int(rot_dir.split("_")[1])

    def discover_rot_dirs(objective: str, dimension: int) -> list[str]:
        base = RESULTS_DIR / objective / f"d{dimension}"
        if not base.exists():
            return []
        return [d.name for d in sorted(base.iterdir()) if (d / "raw.parquet").exists()]

    def load_data(objective: str, dimension: int) -> pd.DataFrame:
        rot_dirs = discover_rot_dirs(objective, dimension)
        dfs = []
        for rot_dir in rot_dirs:
            path = RESULTS_DIR / objective / f"d{dimension}" / rot_dir / "raw.parquet"
            df = pd.read_parquet(path).reset_index()
            df["rot_matrix_idx"] = rot_dir_to_idx(rot_dir)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # discover all (objective, dimension) combos that have results
    _available = set()
    if RESULTS_DIR.exists():
        for _obj_dir in RESULTS_DIR.iterdir():
            for _dim_dir in _obj_dir.iterdir():
                if _dim_dir.name.startswith("d") and _dim_dir.name[1:].isdigit():
                    for _rot_dir in _dim_dir.iterdir():
                        if (_rot_dir / "raw.parquet").exists():
                            _available.add((_obj_dir.name, int(_dim_dir.name[1:])))

    objectives = sorted(set(o for o, _ in _available))
    dimensions = sorted(set(d for _, d in _available))
    return dimensions, discover_rot_dirs, load_data, objectives, rot_dir_to_idx


@app.cell(hide_code=True)
def _(dimensions, mo, objectives):
    objective_sel = mo.ui.dropdown(
        objectives, value=objectives[0] if objectives else None, label="Objective"
    )
    dimension_sel = mo.ui.dropdown(
        [str(d) for d in dimensions],
        value=str(dimensions[0]) if dimensions else None,
        label="Dimension",
    )
    mo.hstack([objective_sel, dimension_sel], gap=2)
    return dimension_sel, objective_sel


@app.cell
def _(dimension_sel, discover_rot_dirs, load_data, mo, objective_sel):
    mo.stop(objective_sel.value is None or dimension_sel.value is None)

    _dim = int(dimension_sel.value)
    rot_dirs_available = discover_rot_dirs(objective_sel.value, _dim)
    data = load_data(objective_sel.value, _dim)
    return data, rot_dirs_available


@app.cell(hide_code=True)
def _(data, mo, rot_dirs_available):
    mo.stop(
        data.empty,
        mo.callout(mo.md("No data found — run the experiment first."), kind="warn"),
    )

    all_cma_evals = sorted(data["cma_num_evaluations"].unique().tolist())
    all_run_ids = sorted(data["run_id"].unique().tolist())

    # default selection: ~8 evenly-spaced snapshots
    _step = max(1, len(all_cma_evals) // 8)
    _default_cma = [str(x) for x in all_cma_evals[::_step]]

    rot_sel = mo.ui.multiselect(
        rot_dirs_available, value=rot_dirs_available, label="Rotation matrices"
    )
    cma_evals_sel = mo.ui.multiselect(
        [str(x) for x in all_cma_evals],
        value=_default_cma,
        label="CMA-ES snapshots (# evaluations)",
    )
    run_id_sel = mo.ui.dropdown(
        ["All"] + [str(r) for r in all_run_ids],
        value="All",
        label="Starting point (run_id)",
    )
    agg_stat_sel = mo.ui.radio(
        ["mean", "median"], value="mean", label="Aggregation statistic"
    )

    mo.vstack(
        [
            mo.hstack([rot_sel, run_id_sel, agg_stat_sel], gap=2),
            cma_evals_sel,
        ],
        gap=1,
    )
    return agg_stat_sel, cma_evals_sel, rot_sel, run_id_sel


@app.cell(hide_code=True)
def _(
    agg_stat_sel,
    aggregate_convergence_series,
    cm,
    cma_evals_sel,
    data,
    dimension_sel,
    mcolors,
    mo,
    objective_sel,
    plt,
    rot_dir_to_idx,
    rot_sel,
    run_id_sel,
):
    mo.stop(not cma_evals_sel.value or not rot_sel.value)

    _selected_cma_evals = sorted(int(x) for x in cma_evals_sel.value)
    _selected_rot_idxs = [rot_dir_to_idx(d) for d in rot_sel.value]
    _conditions = ["cma_mean", "cma_start"]
    _cond_labels = {"cma_mean": "CMA-ES mean as x₀", "cma_start": "Original x₀"}
    _is_individual = run_id_sel.value != "All"
    _stat = agg_stat_sel.value

    _filtered = data[
        data["rot_matrix_idx"].isin(_selected_rot_idxs)
        & data["cma_num_evaluations"].isin(_selected_cma_evals)
    ]

    _norm = mcolors.Normalize(
        vmin=min(_selected_cma_evals), vmax=max(_selected_cma_evals)
    )
    _cmap = cm.plasma

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for _ax, _condition in zip(_axes, _conditions):
        _cond_data = _filtered[_filtered["condition"] == _condition]

        for _cma_eval in _selected_cma_evals:
            _snap = _cond_data[_cond_data["cma_num_evaluations"] == _cma_eval]
            _color = _cmap(_norm(_cma_eval))
            _label = f"{_cma_eval:,}"

            if _is_individual:
                _run_data = _snap[_snap["run_id"] == int(run_id_sel.value)]
                _series = _run_data.set_index("num_evaluations")["best"].sort_index()
                if not _series.empty:
                    _ax.plot(
                        _series.index,
                        _series.values,
                        color=_color,
                        lw=1.2,
                        alpha=0.9,
                        label=_label,
                    )
            else:
                # aggregate over (run_id × rot_matrix_idx)
                _series_list = [
                    _grp.set_index("num_evaluations")["best"].sort_index()
                    for _, _grp in _snap.groupby(["run_id", "rot_matrix_idx"])
                    if not _grp.empty
                ]
                if _series_list:
                    _agg = aggregate_convergence_series(_series_list)
                    _ax.plot(
                        _agg.index, _agg[_stat], color=_color, lw=1.4, label=_label
                    )
                    _ax.fill_between(
                        _agg.index, _agg["q25"], _agg["q75"], color=_color, alpha=0.13
                    )

        _ax.set_xscale("log")
        _ax.set_yscale("log")
        _ax.set_xlabel("Powell evaluations")
        _ax.set_title(_cond_labels[_condition])
        _ax.grid(True, which="both", alpha=0.3)
        _ax.legend(title="CMA-ES evals", fontsize=8, title_fontsize=8)

    _axes[0].set_ylabel("f(x_best)")

    _view = f"run_id = {run_id_sel.value}"
    _fig.suptitle(
        f"Powell + CMA-ES C matrix  ·  {objective_sel.value}, d = {dimension_sel.value}  ·  {_view}",
        y=1.02,
    )
    # plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
