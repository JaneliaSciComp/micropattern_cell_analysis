import marimo

__generated_with = "0.14.15"
app = marimo.App(width="medium", app_title="Micropattern Cell Analysis")


@app.cell
def _():
    import marimo as mo
    import os
    import pathlib
    import nd2
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import skimage
    import xarray as xr
    return Path, mo, nd2, np, plt


@app.cell
def _(Path, mo):
    path_str = "/groups/scicompsoft/home/kittisopikulm/src/micropattern_cell_analysis/valelab/_for_Mark/patterned_data/"
    file_browser = mo.ui.file_browser(
        initial_path=Path(path_str),
        selection_mode = "file",
        multiple=True
    )
    file_browser
    return (file_browser,)


@app.cell
def _():
    # do not worry about clustered and dispersed
    # focus on patterned_data
    # 250710_patterned_plate_9_good, C2, C3, C4
    # Take any nd2 file, do not look at excluded cell folder,
    # cell_filtering.txt is informational only, no need to parse it
    # ignore tile.nd2, those tiles are big
    # only look for nd2 files that start with cell
    # NoV = No virus
    # _for_Mark is a duplicate of raw data
    # D05 was imaged on multiple days
    return


@app.cell
def _(mo):
    mo.md(r"""https://resisted-curiosity-682.notion.site/Final-folder-structure-26879054849480ac8473e5427c072825""")
    return


@app.cell
def _():
    folder_structure = """
    - no TRAK / TRAK1 / TRAK2 - mito
        - 250612_patterned_plate_3 (B02 / B03 / B04)
        - 250710_patterned_plate_9 (C02 (n=8) / C03 (n=8) / C04 (n=11))
        - 250731_patterned_plate_11 (D06 / E05 / F05)
    - no TRAK / TRAK1 / TRAK2 - peroxisome
        - 250606_patterned_plate_2 (G10 / F06 / G06)
        - 250612_patterned_plate_3 (E02 / E03 / E04)
        - 250626_patterned_plate_7 (F02 / G03 / F04)
        - 250710_patterned_plate_9 (D02 / D03 / D04)
    - no TRAK / TRAK1 / TRAK2 - 60mers
        - 250620_patterned_plate_5 (E02 / E04 / E03) - cGP80s
        - 250624_patterned_plate_6 (E07 / E08 / E09) - cGP200s (better)
    - TRAK2: wt / DRH / DRH+Spindly
        - 250606_patterned_plate_2 (C02 / E04 / D05)
        - 250612_patterned_plate_3 (B04 / B07 / B08)
        - 250710_patterned_plate_9 (C04 / C07 / C08)
        - 250731_patterned_plate_11 (F05 / D07 / E07)
    - TRAK1: wt / DRH / DRH+Spindly
        - 250521_patterned_plate_1 (B06 / E06 / D07) - pilot / may not have enough cells
        - 250612_patterned_plate_3 (B03 / B05 / B06)
        - 250710_patterned_plate_9 (C03 / C05 / C06)
        - 250731_patterned_plate_11 (E05 / E06 / F06)
    - TRAK2: wt / S84A / S84E
        - 250606_patterned_plate_2 (C02 / B05 / C05)
        - 250612_patterned_plate_3 (B04 / B09 / B10)
        - 250710_patterned_plate_9 (C04 / C09 / C10)
        - 250731_patterned_plate_11 (F05 / F07 / D08)
    - TRAK2: wt -/+ Ars / S84A -/+ Ars # Christina needs to double check this
        - 250626_patterned_plate_7 (B02 / B03 / B09 / B10) -  pilot / may not have enough
        - 250710_patterned_plate_9 (F02 / F03 / F08 / F09)
        - 250731_patterned_plate_11 (E04 / F04 / B04 / C04)
        - 250807_patterned_plate_12 (F05 / G05 / B05 / C05))
    - wt cells: ctrl siRNA -/+ Ars / MAPK9 siRNA -/+ Ars
        - 250710_patterned_plate_9 (F05 / G03 / F11 / G09)
        - 250724_patterned_plate_10 (D05 / E05 / E02 / B03)
        - 250731_patterned_plate_11 ( E03 / F03 / B03 / C03)
        - 250807_patterned_plate_12 (F04 / G04 / B04 / C04)
    - wt cells: +/- Ars
        - 250618_patterned_plate_4 (B02 / B08)
        - 250624_patterned_plate_6 (B06 / D03)
        - 250701_patterned_plate_8 (B06 / D06)
    """
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    * Metric between clustered and dispersed mitochondira
    * Get probability density distribution between the two
    * Sum across the intensities
    * 60mers do not have a triplicate
    * Just extract the density map
    """
    )
    return


@app.cell
def _(file_browser):
    selection = file_browser.value
    selection[0].name
    return (selection,)


@app.cell
def _(selection):
    selection[0].path
    return


@app.cell
def _(mo):
    mo.md(r"""https://resisted-curiosity-682.notion.site/Micropatterned-cell-analysis-1fc79054849480e887f6d45ba3aeecfb""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    1. File structure:
        - parent folder corresponds to a 96 well plate that was plated and fixed on the same day
            - \\[prfs.hhmi.org](http://prfs.hhmi.org/)\valelab\\Gaby\Vale\imaging\2025\250521_round_E_patterned_1
        - subfolders correspond to individual wells with different conditions (in this case expressing different variants of TRAK); in the name they contain information about the date imaged and the condition
        - each subfolder contains .nd2 stacks corresponding to a cell that was acquired

    2. Data for each cell:
        - 4 colour z-stacks through single patterned cells as .nd2 files
        405 - nuclear stain (Hoechst dye)
        488 - organelle of interest = mitochondria or peroxisomes
        561 - expressed TRAK protein that is expected to affect distribution
        640 - micro pattern visualised by Fibronectin-647
        - we might consider processing by denoising using NIS Elements; this could be very effective to boost our signal:noise ratio
    """
    )
    return


@app.cell
def _():
    data_path = "valelab/Gaby/Vale/imaging/2025/250521_patterned_plate_1"
    return (data_path,)


@app.cell
def _(Path, data_path):
    datasets = [str(d) for d in Path(data_path).iterdir() if d.is_dir()]
    return (datasets,)


@app.cell
def _(datasets, mo):
    dataset_dropdown = mo.ui.dropdown(options=datasets, label="Select Dataset", value=datasets[0])
    dataset_dropdown
    return (dataset_dropdown,)


@app.cell
def _(Path, dataset_dropdown):
    nd2_images = [str(image) for image in Path(dataset_dropdown.selected_key).iterdir() if image.suffix == ".nd2" and image.name.startswith("Cell")]
    return (nd2_images,)


@app.cell
def _(mo, nd2_images):
    images_dropdown = mo.ui.dropdown(options=nd2_images, label = "Select Image", value=nd2_images[0])
    images_dropdown
    return (images_dropdown,)


@app.cell
def _(Path, images_dropdown, nd2):
    image_path = Path(images_dropdown.selected_key)
    image = nd2.imread(image_path, xarray=True, dask=True)
    return (image,)


@app.cell
def _(Path, data_path, nd2):
    cell_1 = nd2.imread(Path(data_path)/"B06_250528_TRAK1-wt/Cell1.nd2", xarray=True)
    return


@app.cell
def _(np):
    def scale(arr):
        min = np.min(arr)
        max = np.max(arr)
        return (arr - min)/(max-min)
    return (scale,)


@app.cell
def _(image, mo):
    channel_dropdown = mo.ui.dropdown(
        options=[str(c) for c in image.C.values],
        value=image.C.values[0],
        label="Channel"
    )
    channel_dropdown
    return (channel_dropdown,)


@app.cell
def _(image):
    image
    return


@app.cell
def _(image, mo):
    z_slider = mo.ui.slider(steps=image.Z.values, full_width=True, label="Z")
    return (z_slider,)


@app.cell
def _(mo):
    image_scale_slider = mo.ui.range_slider(
        orientation="vertical",
        start=0.0,
        stop=1.0,
        step=0.01,
        full_width=True,
        show_value=True)
    return (image_scale_slider,)


@app.cell
def _(channel_dropdown, image, z_slider):
    image_CZ = image.sel(C=channel_dropdown.selected_key, Z=z_slider.value)
    return (image_CZ,)


@app.cell
def _(image_CZ, image_scale_slider, plt, scale):
    def imshow_cz():
        plt.imshow(
            scale(image_CZ),
            vmin=image_scale_slider.value[0],
            vmax=image_scale_slider.value[1]
        )
        #plt.scatter(centroid[0], centroid[1], color='red', marker='x')
        return plt.gca()

    return (imshow_cz,)


@app.cell
def _(
    channel_dropdown,
    dataset_dropdown,
    image_scale_slider,
    images_dropdown,
    imshow_cz,
    mo,
    z_slider,
):
    mo.vstack([
        dataset_dropdown,
        images_dropdown,
        mo.hstack([
            imshow_cz(),
            image_scale_slider
        ]),
        mo.hstack([channel_dropdown,z_slider])
    ])
    return


@app.cell
def _(mean_values, polar_bar):
    polar_bar(mean_values)
    return


@app.cell
def _(mean_values, np, plt):
    plt.plot(np.linspace(0, 360, 37)[0:36], mean_values)
    return


@app.cell
def _(pattern_mip_scaled, plt, rp):
    plt.imshow(pattern_mip_scaled, vmax=0.2)
    plt.scatter(rp[0].centroid[0], rp[0].centroid[1], color='red', marker='x')
    return


if __name__ == "__main__":
    app.run()
