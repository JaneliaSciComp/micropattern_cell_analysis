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

def get_image_hat(img):
    img_template_sum_projection = np.sum(img.sel(C="640").isel(C=0), axis=0)
    assert img_template_sum_projection.ndim == 2
    img_template_sum_projection_norm = img_template_sum_projection / np.max(img_template_sum_projection)
    #img_template_sum_projection_hat = np.abs(np.fft.fft2(img_template_sum_projection))
    img_template_sum_projection_norm_2048 = img_template_sum_projection_norm[:2048,:2048]
    img_template_sum_projection_norm_2048_hat = np.fft.fft2(img_template_sum_projection_norm_2048)
    return img_template_sum_projection_norm_2048_hat

def match_template(img, *, template_hat = None):
    if isinstance(img, str) or isinstance(img, pathlib.Path):
        img_path = img
        img = nd2.imread(img_path, xarray=True)
    if template_hat is None:
        template_hat = get_template_hat(1326)
    img_template_hat = get_image_hat(img)
    template_matching = np.fft.fftshift(np.real(np.fft.ifft2(template_hat * img_template_hat)))
    return template_matching

def max_match_template(img, *, template_hat = None):
    template_matching = match_template(img, template_hat = template_hat)
    max_idx = np.argmax(template_matching)
    return np.unravel_index(max_idx, template_matching.shape)


def score_template_match(img_path, *, template_hat = None, template = None):
    # Load image
    img = nd2.imread(img_path, xarray=True)
    sumproj = np.sum(img[:,1,:2048,:2048], axis=0) > 6100

    # Match
    max_coords = max_match_template(img, template_hat=template_hat)
    shifted_template = np.roll(template, (max_coords[0] - 1024, max_coords[1] - 1024), axis=(0,1))
    score = np.sum(sumproj & shifted_template)/(np.sum(shifted_template > 0))
    score = score.values.item()

    # Plot figure
    fig = plt.figure()
    plt.imshow(shifted_template)
    plt.imshow(sumproj, alpha=0.5)
    plt.scatter(max_coords[1], max_coords[0])
    plt.annotate(text="{:.3f}%".format(score*100), xy=(max_coords[1] + 100, max_coords[0]), color="yellow")
    plt.title(img_path, loc="right")
    pdf_path = pathlib.Path("template_matching",*pathlib.Path(img_path).parts[-3:]).with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pdf_path)

    return score


def main(root_path):
    #Set up template
    template_hat = get_template_hat(1326)
    template = get_padded_template_at_width(1326)

    print(f"Scanning {root_path}")

    # img_path = "/groups/vale/valelab/_for_Mark/patterned_data/250521_patterned_plate_1/B06_250528_TRAK1-wt/Cell8.nd2"
    for (dirpath, dirnames, filenames) in pathlib.Path(root_path).walk():
        if dirpath.parts[-1] == "MaxIP" or dirpath.parts[-1] == "MaxIPs":
            continue
        img_paths = []
        scores = []
        csv_path = pathlib.Path("template_matching", *dirpath.parts[-2:], "template_matching.csv")
        xlsx_path = pathlib.Path("template_matching", *dirpath.parts[-2:], "template_matching.xlsx")
        print(csv_path)
        print(xlsx_path)
        for filename in filenames:
            if filename.endswith(".nd2") and (filename.startswith("Cell") or filename.startswith("cell")):
                img_path = dirpath / filename
                img_paths.append(img_path)
                try:
                    print(img_path)
                    score = score_template_match(img_path, template_hat=template_hat, template=template)
                    print(score)
                    scores.append(score)
                except Exception as e:
                    scores.append(float('nan'))
                    print(f"An error occurred with {img_path}: {e}")
                    traceback.print_exc()

        print("Writing data frame")
        print(f"{img_paths=}")
        print(f"{scores=}")

        score_df = pl.DataFrame(
                {
                    "path": [str(img_path) for img_path in img_paths],
                    "template_matching_score": scores
                },
                {
                    "path": pl.datatypes.String,
                    "template_matching_score": pl.datatypes.Float64
                }
        )
        print(score_df)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        score_df.write_csv(csv_path)
        score_df.write_excel(xlsx_path)

if __name__ == "__main__":
    main(sys.argv[1])
