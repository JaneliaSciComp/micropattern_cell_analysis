import marimo as mo
import cairosvg
import skimage
import numpy as np
import pymupdf
import io
import nd2
import matplotlib.pyplot as plt
import pathlib
import polars as pl
import sys
import traceback
import netCDF4
from scipy.ndimage import distance_transform_edt
from matplotlib.backends.backend_pdf import PdfPages

def get_template_at_width(width):
    file = pymupdf.open("single_pattern.ai")
    png_bytes = cairosvg.svg2png(file[0].get_svg_image(), output_width=width, output_height=width)
    with io.BytesIO() as buf:
        buf.write(png_bytes)
        template_img = skimage.io.imread(buf)
    return template_img


def get_padded_template_at_width(template_width, *, base_template=None):
    if base_template is None:
        base_template = get_template_at_width(template_width)[:,:,0]
    pad = (2048-template_width)//2
    template = np.pad(base_template,(pad, pad))
    return template

def get_template_hat(template_width):
    """
    get frequency space template at size 2048
    """
    template = get_padded_template_at_width(template_width)
    #pad = (2048-template_width)//2
    #template = np.pad(get_template_at_width(template_width)[:,:,0],(pad, pad))
    template_flipped = np.flip(template, axis=(0,1))
    template_hat = np.fft.fft2(template_flipped)
    return template_hat

offset_overrides = {
    "/groups/vale/valelab/_for_Mark/patterned_data/250731_patterned_plate_11_good/E05_250808_TRAK1_wt/Cell2.nd2": [256,128],
    "/groups/vale/valelab/_for_Mark/patterned_data/250731_patterned_plate_11_good/E05_250808_TRAK1_wt/Cell2 - Denoised.nd2": [256,128],
    "/groups/vale/valelab/_for_Mark/patterned_data/250731_patterned_plate_11_good/E05_250808_TRAK1_wt/Cell5.nd2": [204,128],
    "/groups/vale/valelab/_for_Mark/patterned_data/250731_patterned_plate_11_good/E05_250808_TRAK1_wt/Cell5 - Denoised.nd2": [204,128],

    "/groups/vale/valelab/_for_Mark/patterned_data/250731_patterned_plate_11_good/E05_250808_TRAK1_wt/Cell8.nd2": [256,128],
    "/groups/vale/valelab/_for_Mark/patterned_data/250731_patterned_plate_11_good/E05_250808_TRAK1_wt/Cell8 - Denoised.nd2": [256,128],
    "/groups/vale/valelab/_for_Mark/patterned_data/250612_patterned_plate_3/B06_250617_TRAK1_mDRH_dSp/Cell12.nd2": [64,128]
}

def get_image_hat(img, offset=[128, 128]):
    channel_640 = img.sel(C="640")
    if channel_640.ndim > 3:
        # If there are two channel 640s, then use the first one
        # TODO: Make this more specific to the channels
        channel_640 = channel_640.isel(C=0)
    img_template_sum_projection = np.sum(img.sel(C="640"), axis=0)
    assert img_template_sum_projection.ndim == 2
    img_template_sum_projection_norm = img_template_sum_projection / np.max(img_template_sum_projection)
    #img_template_sum_projection_hat = np.abs(np.fft.fft2(img_template_sum_projection))
    img_template_sum_projection_norm_2048 = img_template_sum_projection_norm[offset[0]:2048+offset[0],offset[1]:2048+offset[1]]
    img_template_sum_projection_norm_2048_hat = np.fft.fft2(img_template_sum_projection_norm_2048)
    return img_template_sum_projection_norm_2048_hat

"""
def get_image_hat(img):
    channel_640 = img.sel(C="640")
    if channel_640.ndim > 3:
        # If there are two channel 640s, then use the first one
        # TODO: Make this more specific to the channels
        channel_640 = channel_640.isel(C=0)
    img_template_sum_projection = np.sum(channel_640, axis=0)
    assert img_template_sum_projection.ndim == 2
    img_template_sum_projection_norm = img_template_sum_projection / np.max(img_template_sum_projection)
    #img_template_sum_projection_hat = np.abs(np.fft.fft2(img_template_sum_projection))
    img_template_sum_projection_norm_2048 = img_template_sum_projection_norm[:2048,:2048]
    img_template_sum_projection_norm_2048_hat = np.fft.fft2(img_template_sum_projection_norm_2048)
    return img_template_sum_projection_norm_2048_hat
"""

def match_template(img, *, template_hat = None, offset=None):
    if isinstance(img, str) or isinstance(img, pathlib.Path):
        img_path = img
        offset = offset_overrides.get(img_path, [128, 128])
        img = nd2.imread(img_path, xarray=True)
    if template_hat is None:
        template_hat = get_template_hat(1326)
    img_template_hat = get_image_hat(img, offset=offset)
    template_matching = np.fft.fftshift(np.real(np.fft.ifft2(template_hat * img_template_hat)))
    return template_matching

roi_overrides = {
    "/groups/vale/valelab/_for_Mark/patterned_data/250731_patterned_plate_11_good/F06_250811_TRAK1_mDRH_dSp/Cell3.nd2": [slice(None), slice(0,1200)]
}

def max_match_template(img, *, template_hat = None, offset=None, roi=None):
    template_matching = match_template(img, template_hat = template_hat, offset=offset)

    if roi is not None:
       template_matching = template_matching[*roi]

    max_idx = np.argmax(template_matching)
    out = np.unravel_index(max_idx, template_matching.shape)

    if roi is not None:
        # reshift based on start
        if roi[0].start is not None:
            out = (out[0] + roi[0].start, out[1])
        if roi[1].start is not None:
            out = (out[0], out[1] + roi[1].start)

    return out

def stretch01(img, *, min_percentile=0.1, max_percentile=99.9):
    _min = np.percentile(img, min_percentile)
    _max = np.percentile(img, max_percentile)
    return np.clip((img - _min)/(_max - _min), 0, 1)

def make_rgb(R, G, B):
    RGB = np.zeros([3, *R.shape[-2:]], dtype="float32")
    if R is not None:
        RGB[0,:,:] = R
    if G is not None:
        RGB[1,:,:] = G
    if B is not None:
        RGB[2,:,:] = B
    return np.permute_dims(RGB,(1,2,0))

def draw_scale_bar(pixel_length):
    plt.plot([800, 800+pixel_length], [900, 900], color="white")
    plt.text(790, 950, "5 μm", color="white")

def score_template_match(img_path, *, template_hat = None, template = None):
    # Load image
    img = nd2.imread(img_path, xarray=True)
    sumproj = np.sum(img[:,1,128:2048+128,128:2048+128], axis=0)
    sumproj_threshold = skimage.filters.threshold_otsu(sumproj.to_numpy())
    sumproj_thresholded = sumproj > sumproj_threshold

    # Match
    offset = offset_overrides.get(str(img_path), [128, 128])
    roi = roi_overrides.get(str(img_path), None)
    print(f"{img_path = }")
    print(f"{offset = }")
    print(f"{roi = }")
    max_coords = max_match_template(img, template_hat=template_hat, offset=offset, roi=roi)
    shifted_template = np.roll(template, (max_coords[0] - 1024, max_coords[1] - 1024), axis=(0,1))
    shifted_template_contour = skimage.measure.find_contours(shifted_template)
    score = np.sum(sumproj_thresholded & shifted_template)/(np.sum(shifted_template > 0))
    score = score.values.item()

    cropped_proj_img = np.sum(img[:,:,max_coords[0]-512+128:max_coords[0]+512+128, max_coords[1]-512+128:max_coords[1]+512+128], axis=0)
    cropped_template_contour = shifted_template_contour[0].copy()
    cropped_template_contour[:,0] -= (max_coords[0]-512)
    cropped_template_contour[:,1] -= (max_coords[1]-512)

    cropped_nuc_proj = cropped_proj_img.sel(C="405")
    nuc_proj_threshold = skimage.filters.threshold_otsu(cropped_nuc_proj.to_numpy())
    cropped_nuc_mask = cropped_nuc_proj > nuc_proj_threshold
    cropped_nuc_label = skimage.measure.label(cropped_nuc_mask)
    cropped_nuc_props = skimage.measure.regionprops(cropped_nuc_label)
    cropped_nuc_max_area = np.argmax([p.area for p in cropped_nuc_props])
    cropped_nuc_mask = (cropped_nuc_label == cropped_nuc_max_area+1)
    cropped_nuc_edt = distance_transform_edt(np.invert(cropped_nuc_mask))

    top_arch_mask = np.zeros_like(cropped_nuc_mask)
    top_arch_mask[
        np.round(cropped_template_contour[1083:1951,0]).astype("int"),
        np.round(cropped_template_contour[1083:1951,1]).astype("int")
    ] = True
    cropped_arch_edt = distance_transform_edt(np.invert(top_arch_mask))

    acute_arch_mask = np.zeros_like(cropped_nuc_mask)
    acute_arch_mask[
        np.round(cropped_template_contour[1300:1734,0]).astype("int"),
        np.round(cropped_template_contour[1300:1734,1]).astype("int")
    ] = True
    cropped_acute_arch_edt = distance_transform_edt(np.invert(acute_arch_mask))

    metadata = img.metadata["metadata"]
    lateral_pixel_pitch = metadata.channels[0].volume.axesCalibration[0]
    perinuclear_space_distance_um = 5 # micrometers
    perinuclear_space_distance_pixels = perinuclear_space_distance_um/lateral_pixel_pitch

    cropped_proj_mitochondria = cropped_proj_img.sel(C="488")
    cropped_proj_mitochondria_stretched = stretch01(cropped_proj_mitochondria)

    # Assume the left and right edges consist of background
    cropped_background = np.concatenate((cropped_proj_mitochondria_stretched[:,:128], cropped_proj_mitochondria_stretched[:,-128:]), axis=1)
    cropped_background_threshold = np.percentile(cropped_background, 99.99)
    cropped_proj_mitochondria_bg_subtracted = cropped_proj_mitochondria_stretched - cropped_background_threshold
    # Set negative values to 0
    cropped_proj_mitochondria_bg_subtracted = np.clip(cropped_proj_mitochondria_bg_subtracted, 0, None)

    perinuclear_mask = cropped_nuc_edt < perinuclear_space_distance_pixels
    peripheral_mask = cropped_arch_edt <= cropped_nuc_edt
    peripheral_mask &= np.invert(perinuclear_mask)
    peripheral_5um_mask  = (cropped_arch_edt <= perinuclear_space_distance_pixels) & peripheral_mask
    peripheral_mask     &=  cropped_arch_edt <= perinuclear_space_distance_pixels * 1.75

    acute_peripheral_mask = cropped_acute_arch_edt <= cropped_nuc_edt
    acute_peripheral_mask &= np.invert(perinuclear_mask)
    acute_peripheral_mask &=  cropped_arch_edt <= perinuclear_space_distance_pixels

    mitochondria_sum = np.sum(cropped_proj_mitochondria_bg_subtracted)

    perinuclear_mitochondria = perinuclear_mask * cropped_proj_mitochondria_bg_subtracted
    peripheral_mitochondria  = peripheral_mask  * cropped_proj_mitochondria_bg_subtracted
    peripheral_5um_mitochondria = peripheral_5um_mask * cropped_proj_mitochondria_bg_subtracted
    acute_peripheral_mitochondria = acute_peripheral_mask * cropped_proj_mitochondria_bg_subtracted

    peripheral_contour = skimage.measure.find_contours(peripheral_mask)[0]
    peripheral_5um_contour = skimage.measure.find_contours(peripheral_5um_mask)[0]
    acute_peripheral_contour = skimage.measure.find_contours(acute_peripheral_mask)[0]
    perinuclear_contour = skimage.measure.find_contours(perinuclear_mask)[0]

    perinuclear_sum = np.sum(perinuclear_mitochondria)

    peripheral_sum = np.sum(peripheral_mitochondria)
    peripheral_5um_sum = np.sum(peripheral_5um_mitochondria)
    acute_peripheral_sum = np.sum(acute_peripheral_mitochondria)

    pp_sum = perinuclear_sum + peripheral_sum
    peripheral_percent = peripheral_sum / pp_sum * 100
    peripheral_5um_percent = peripheral_5um_sum / (peripheral_5um_sum + perinuclear_sum) * 100
    acute_peripheral_percent = acute_peripheral_sum / (acute_peripheral_sum + perinuclear_sum) * 100


    # draw contour around 0.5 um of nucleus
    cropped_nuclear_contour = get_nuclear_contour(cropped_nuc_edt < 0.5/lateral_pixel_pitch)

    # Plot figure to PDF
    relative_path = pathlib.Path(img_path).relative_to("/groups/vale/valelab/_for_Mark/patterned_data")
    pdf_path = pathlib.Path("template_matching",*relative_path.parts).with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        # Template matching figure
        fig = plt.figure()
        #plt.imshow(shifted_template)
        #plt.imshow(sumproj_thresholded) # , alpha=0.5)
        plt.imshow(stretch01(sumproj))
        plt.plot(shifted_template_contour[0][:,1], shifted_template_contour[0][:,0], color="black")
        plt.scatter(max_coords[1], max_coords[0])
        plt.annotate(text="{:.3f}%".format(score*100), xy=(max_coords[1] + 100, max_coords[0]), color="yellow")
        plt.title(img_path, loc="right")
        pdf.savefig()
        plt.close()

        fig = plt.figure()
        cropped_rgb = make_rgb(
           stretch01(cropped_proj_img.sel(C="488")),
           None,
           stretch01(cropped_proj_img.sel(C="405"))
        )
        plt.imshow(cropped_rgb)
        plt.plot(cropped_template_contour[:,1], cropped_template_contour[:,0], color="white")
        # Top Arch
        plt.plot(cropped_template_contour[1083:1951,1], cropped_template_contour[1083:1951,0], color="magenta")
        # Top Left point
        plt.scatter(cropped_template_contour[1083,1], cropped_template_contour[1083,0]+60, color="white")
        # Top Right point
        plt.scatter(cropped_template_contour[1951,1], cropped_template_contour[1951,0]+60, color="white")
        # Bottom Middle point
        plt.scatter(cropped_template_contour[12,1], cropped_template_contour[12,0], color="white")
        # Bottom Left point
        plt.scatter(cropped_template_contour[1083,1], cropped_template_contour[12,0], color="white")
        # Bottom Right point
        plt.scatter(cropped_template_contour[1951,1], cropped_template_contour[12,0], color="white")
        draw_scale_bar(perinuclear_space_distance_pixels)
        pdf.savefig()
        plt.close()

        fig = plt.figure()
        plt.imshow(make_rgb(
            stretch01(-cropped_arch_edt),
            stretch01(-cropped_nuc_edt),
            stretch01(-cropped_arch_edt)
        ))
        plt.plot(cropped_template_contour[1083:1951,1],cropped_template_contour[1083:1951,0], color="black")
        draw_scale_bar(perinuclear_space_distance_pixels)
        pdf.savefig()
        plt.close()

        fig = plt.figure()
        plt.imshow(cropped_arch_edt <= cropped_nuc_edt)
        plt.plot(cropped_template_contour[:,1], cropped_template_contour[:,0], color="white")
        plt.plot(cropped_template_contour[1083:1951,1],cropped_template_contour[1083:1951,0], color="magenta", alpha=0.5)
        plt.plot(cropped_nuclear_contour[:,1], cropped_nuclear_contour[:,0], color="blue", alpha=0.5)
        draw_scale_bar(perinuclear_space_distance_pixels)
        pdf.savefig()
        plt.close()

        fig = plt.figure()
        plt.imshow(cropped_rgb)
        plt.plot(peripheral_contour[:,1],  peripheral_contour[:,0], color="yellow", linestyle="dotted")
        plt.plot(peripheral_5um_contour[:,1],  peripheral_5um_contour[:,0], color="yellow")
        plt.plot(acute_peripheral_contour[:,1],  acute_peripheral_contour[:,0], color="yellow", linestyle="dotted")
        plt.plot(cropped_template_contour[1083:1951,1],cropped_template_contour[1083:1951,0], color="magenta", alpha=0.5)
        plt.plot(cropped_template_contour[1300:1734,1],cropped_template_contour[1300:1734,0], color="cyan", alpha=0.5)
        plt.plot(perinuclear_contour[:,1], perinuclear_contour[:,0], color="blue")
        pdf.savefig()
        plt.close()

        fig = plt.figure()
        # Draw peripheral mitochondria as yellow
        plt.imshow(make_rgb(
            peripheral_5um_mitochondria,
            peripheral_5um_mitochondria,
            perinuclear_mitochondria
        ))
        plt.plot(cropped_template_contour[1083:1951,1],cropped_template_contour[1083:1951,0], color="white", alpha=0.5)
        plt.plot(cropped_template_contour[1300:1734,1],cropped_template_contour[1300:1734,0], color="cyan", alpha=0.5)
        plt.plot(cropped_nuclear_contour[:,1], cropped_nuclear_contour[:,0], color="white", alpha=0.5)
        plt.title("P5um/(P5um+N): {:.1f}%, P5um/Crop: {:.1f}%, N/Crop: {:.1f}%".format(peripheral_5um_percent, peripheral_5um_sum/mitochondria_sum*100, perinuclear_sum/mitochondria_sum*100))
        draw_scale_bar(perinuclear_space_distance_pixels)
        pdf.savefig()
        plt.close()

    output = {
            "score": score,
            "perinuclear_sum": perinuclear_sum,
            "peripheral_sum": peripheral_sum,
            "peripheral_5um_sum": peripheral_5um_sum,
            "mitochondria_sum": mitochondria_sum,
            "peripheral_percent": peripheral_percent,
            "peripheral_5um_percent": peripheral_5um_percent,
            "acute_peripheral_percent": acute_peripheral_percent,
    }

    return output

def get_nuclear_contour(nuclear_mask):
    nuclear_contours = skimage.measure.find_contours(nuclear_mask)
    nuclear_contour_index = np.argmax([len(contour) for contour in nuclear_contours])
    return nuclear_contours[nuclear_contour_index]

def main(root_path):
    pl.Config.set_tbl_cell_alignment("RIGHT")

    #Set up template
    template_hat = get_template_hat(1326)
    template = get_padded_template_at_width(1326)

    print(f"Scanning {root_path}")

    # img_path = "/groups/vale/valelab/_for_Mark/patterned_data/250521_patterned_plate_1/B06_250528_TRAK1-wt/Cell8.nd2"
    for (dirpath, dirnames, filenames) in pathlib.Path(root_path).walk():
        if dirpath.parts[-1] == "MaxIP" or dirpath.parts[-1] == "MaxIPs":
            continue
        if dirpath.parts[-1] == "Excluded_cells":
            continue
        img_paths = []
        scores = []
        peripheral_percent = []
        peripheral_5um_percent = []
        acute_peripheral_percent = []
        mitochondria_sum = []
        peripheral_sum = []
        perinuclear_sum = []
        peripheral_5um_sum = []
        relative_path = dirpath.relative_to("/groups/vale/valelab/_for_Mark/patterned_data")
        csv_path = pathlib.Path("template_matching", *relative_path.parts, "template_matching.csv")
        xlsx_path = pathlib.Path("template_matching", *relative_path.parts, "template_matching.xlsx")
        print(csv_path)
        print(xlsx_path)
        for filename in filenames:
            if filename.endswith(".nd2") and (filename.startswith("Cell") or filename.startswith("cell")):
                img_path = dirpath / filename
                img_paths.append(img_path)
                try:
                    print(img_path)
                    output = score_template_match(img_path, template_hat=template_hat, template=template)
                    print(output["score"])
                    scores.append(output["score"])
                    peripheral_percent.append(output["peripheral_percent"])
                    peripheral_5um_percent.append(output["peripheral_5um_percent"])
                    acute_peripheral_percent.append(output["acute_peripheral_percent"])
                    mitochondria_sum.append(output["mitochondria_sum"])
                    peripheral_sum.append(output["peripheral_sum"])
                    peripheral_5um_sum.append(output["peripheral_5um_sum"])
                    perinuclear_sum.append(output["perinuclear_sum"])
                except Exception as e:
                    scores.append(float('nan'))
                    peripheral_percent.append(float('nan'))
                    peripheral_5um_percent.append(float('nan'))
                    acute_peripheral_percent.append(float('nan'))
                    mitochondria_sum.append(float('nan'))
                    peripheral_sum.append(float('nan'))
                    peripheral_5um_sum.append(float('nan'))
                    perinuclear_sum.append(float('nan'))
                    print(f"An error occurred with {img_path}: {e}")
                    traceback.print_exc()

        print("Writing data frame")
        print(f"{img_paths=}")
        print(f"{scores=}")

        score_df = pl.DataFrame(
                {
                    "path": [str(img_path) for img_path in img_paths],
                    "template_matching_score": scores,
                    #"peripheral_percent": peripheral_percent,
                    "peripheral_5um_percent": peripheral_5um_percent,
                    #"acute_peripheral_percent": acute_peripheral_percent,
                    "mitochondria_sum": mitochondria_sum,
                    #"peripheral_sum": peripheral_sum,
                    "peripheral_5um_sum": peripheral_5um_sum,
                    "perinuclear_sum": perinuclear_sum,
                    "peripheral_5um_percent_total": np.array(peripheral_5um_sum) / np.array(mitochondria_sum) * 100,
                    "perinuclear_percent_total": np.array(perinuclear_sum) / np.array(mitochondria_sum) * 100
                },
                {
                    "path": pl.datatypes.String,
                    "template_matching_score": pl.datatypes.Float64,
                    #"peripheral_percent": pl.datatypes.Float64,
                    "peripheral_5um_percent": pl.datatypes.Float64,
                    #"acute_peripheral_percent": pl.datatypes.Float64,
                    "mitochondria_sum": pl.datatypes.Float64,
                    #"peripheral_sum": pl.datatypes.Float64,
                    "peripheral_5um_sum": pl.datatypes.Float64,
                    "perinuclear_sum": pl.datatypes.Float64,
                    "peripheral_5um_percent_total": pl.datatypes.Float64,
                    "perinuclear_percent_total": pl.datatypes.Float64
                }
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        score_df.write_csv(csv_path)
        score_df.write_excel(xlsx_path)

        def right_abbreviate(text: str, max_len: int = 20) -> str:
            if len(text) > max_len:
                return "..." + text[-(max_len - 3):] # Keep the end of the string
            return text

        score_df_abbreviated = score_df.with_columns(
            pl.col("path").map_elements(right_abbreviate).alias("path")
        )
        print(score_df_abbreviated)


if __name__ == "__main__":
    main(sys.argv[1])
