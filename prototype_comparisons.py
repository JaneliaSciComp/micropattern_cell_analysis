import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    import skimage.io
    from pathlib import Path
    from comparison_loader import (
        load_comparisons,
        find_well_directory,
        list_cells,
        load_cell,
        gather_datasets,
        aggregate_datasets,
        sum_channel,
        convert_to_uint16
    )
    return (
        aggregate_datasets,
        convert_to_uint16,
        gather_datasets,
        load_cell,
        load_comparisons,
        mo,
        plt,
        sum_channel,
    )


@app.cell
def _(load_comparisons):
    dfs = load_comparisons()
    return (dfs,)


@app.cell
def _(dfs, mo):
    sheet_selector = mo.ui.dropdown(options=list(dfs.keys()), label="Select Sheet", value=list(dfs.keys())[0] if dfs else None)
    sheet_selector
    return (sheet_selector,)


@app.cell
def _(dfs, sheet_selector):
    selected_df = dfs[sheet_selector.value] if sheet_selector.value else None

    if selected_df is not None:
        # The first column is the plate name (often unnamed)
        plate_col = selected_df.columns[0]
        plates = selected_df[plate_col].to_list()
        # Other columns are conditions
        conditions = selected_df.columns[1:]
    else:
        plates = []
        conditions = []
        plate_col = None
    return conditions, plate_col, selected_df


@app.cell
def _(conditions, mo):
    condition_selector = mo.ui.dropdown(options=conditions, label="Select Condition")
    denoised_toggle = mo.ui.checkbox(label="Denoised", value=False)
    run_button = mo.ui.run_button(label="Aggregate Images")

    mo.hstack([condition_selector, denoised_toggle, run_button])
    return condition_selector, denoised_toggle, run_button


@app.cell
def _(
    aggregate_datasets,
    condition_selector,
    denoised_toggle,
    gather_datasets,
    mo,
    plate_col,
    run_button,
    selected_df,
):
    mo.stop(not run_button.value or not condition_selector.value)

    all_datasets = gather_datasets(selected_df, plate_col, condition_selector.value, denoised_toggle.value)

    aggregated_ds = aggregate_datasets(all_datasets)

    if aggregated_ds is None:
        mo.output.replace(mo.md("No cells found for this condition."))
    else:
        mo.output.replace(mo.md(f"**Aggregated {len(all_datasets)} cells.**"))
    return (all_datasets,)


@app.cell
def _(all_datasets, sum_channel):
    ch488 = sum_channel(all_datasets, "488")
    return (ch488,)


@app.cell
def _(ch488, plt):
    plt.imshow(ch488)
    return


@app.cell
def _(ch488, convert_to_uint16, plt):
    ch488_uint16 = convert_to_uint16(ch488)
    plt.imshow(ch488_uint16)
    return


@app.cell
def _(cell_selector, load_cell, mo):
    mo.stop(not cell_selector or not cell_selector.value)

    ds = load_cell(cell_selector.value)

    # Coordinates for channels
    channels = ds.coords['C'].values.tolist()
    channel_selector = mo.ui.dropdown(options=channels, label="Select Channel", value=channels[0] if channels else None)

    channel_selector
    return channel_selector, ds


@app.cell
def _(channel_selector, ds, mo, plt):
    mo.stop(not channel_selector.value)

    # The variable name in these NetCDFs is often '__xarray_dataarray_variable__'
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].sel(C=channel_selector.value)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data.values, cmap='gray')
    ax.set_title(f"{channel_selector.value}")
    plt.colorbar(im, ax=ax)

    mo.as_html(fig)
    return


if __name__ == "__main__":
    app.run()
