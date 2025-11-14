import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


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
    import pymupdf
    import base64
    import cairo
    import io
    return base64, io, mo, nd2, np, plt, pymupdf, skimage


@app.cell
def _():
    import cairosvg
    import PIL
    from PIL import Image
    return (cairosvg,)


@app.cell
def _():
    #import pymupdf
    #import matplotlib.pyplot as plt
    return


@app.cell
def _(pymupdf):
    file = pymupdf.open("single_pattern.ai")
    return (file,)


@app.cell
def _(file):
    len(file)
    return


@app.cell
def _(file):
    with open("test.svg", "w") as f:
        f.write(file[0].get_svg_image())
    return


@app.cell
def _(img, plt):
    plt.imshow(img[20, 1, :, :])
    return


@app.cell
def _(base64, mo):
    #import marimo as mo
    #import base64

    # Your SVG string
    svg_string = """
    <svg width="100" height="100">
      <circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="red" />
    </svg>
    """

    # 1. Encode the string to bytes
    svg_bytes = svg_string.encode("utf-8")

    # 2. Base64 encode the bytes
    b64_bytes = base64.b64encode(svg_bytes)

    # 3. Decode back to a string for the URI
    b64_string = b64_bytes.decode("utf-8")

    # 4. Create the full Data URI
    data_uri = f"data:image/svg+xml;base64,{b64_string}"

    # Display using mo.image
    mo.image(src=data_uri, alt="Red Circle SVG", width=100)
    return


@app.cell
def _(mo):
    width_slider = mo.ui.slider(steps=range(128,1025))
    width_slider
    return (width_slider,)


@app.cell
def _(mo, width_slider):
    mo.vstack([
        mo.image("test.svg", width=str(width_slider.value)),
        width_slider
    ])
    return


@app.cell
def _():
    img_path = "/groups/vale/valelab/_for_Mark/patterned_data/250521_patterned_plate_1/B06_250528_TRAK1-wt/Cell10.nd2"
    return (img_path,)


@app.cell
def _():
    #import nd2
    return


@app.cell
def _(img_path, nd2):
    img = nd2.imread(img_path)
    return (img,)


@app.cell
def _(img):
    img.shape
    return


@app.cell
def _(plt):

    plt
    return


@app.cell
def _(img):
    img
    return


@app.cell
def _(cairosvg, file):
    png_bytes = cairosvg.svg2png(file[0].get_svg_image(), output_width=256)
    return (png_bytes,)


@app.cell
def _(png_bytes):
    png_bytes
    return


@app.cell
def _(io, png_bytes, skimage):
    buf = io.BytesIO()
    buf.write(png_bytes)
    template_img = skimage.io.imread(buf)
    return (template_img,)


@app.cell
def _(plt, template_img):
    plt.imshow(template_img)
    return


@app.cell
def _(cairosvg, file, io, skimage):
    def get_template_at_width(width):
        png_bytes = cairosvg.svg2png(file[0].get_svg_image(), output_width=width, output_height=width)
        with io.BytesIO() as buf:
            buf.write(png_bytes)
            template_img = skimage.io.imread(buf)
        return template_img
    return (get_template_at_width,)


@app.cell
def _(get_template_at_width, plt):
    plt.imshow(get_template_at_width(2048))
    return


@app.cell
def _(get_template_at_width, np, plt):
    template_hat_abs = np.abs(np.fft.fft2(get_template_at_width(16)[:,:,0]))
    plt.imshow(template_hat_abs / np.max(template_hat_abs))
    return


@app.cell
def _(get_template_at_width, plt):
    plt.imshow(get_template_at_width(64)[:,:,0])
    return


@app.cell
def _(get_template_at_width):
    get_template_at_width(64)[:,:,0]
    return


@app.cell
def _(img, np):
    img_template_sum_projection = np.sum(img[:,1,:,:], axis=0)
    img_template_sum_projection_norm = img_template_sum_projection / np.max(img_template_sum_projection)
    return img_template_sum_projection, img_template_sum_projection_norm


@app.cell
def _(img_template_sum_projection):
    img_template_sum_projection
    return


@app.cell
def _(img_template_sum_projection, np, plt):
    img_template_sum_projection_hat = np.abs(np.fft.fft2(img_template_sum_projection))
    plt.imshow(np.fft.fftshift(np.log(img_template_sum_projection_hat / np.max(img_template_sum_projection_hat))))
    return


@app.cell
def _(img_template_sum_projection_norm):
    img_template_sum_projection_norm_2048 = img_template_sum_projection_norm[:2048,:2048]
    return (img_template_sum_projection_norm_2048,)


@app.cell
def _(img_template_sum_projection_norm_2048, np):
    img_template_sum_projection_norm_2048_hat = np.fft.fft2(img_template_sum_projection_norm_2048)
    return (img_template_sum_projection_norm_2048_hat,)


@app.cell
def _(img_template_sum_projection_norm_2048_hat, np, plt):
    plt.imshow(np.abs(np.fft.ifft2(img_template_sum_projection_norm_2048_hat)))
    return


@app.cell
def _(get_template_at_width, np):
    def get_template_hat(template_width):
        """
        get frequency space template at size 2048
        """
        pad = (2048-template_width)//2
        template = np.pad(get_template_at_width(template_width)[:,:,0],(pad, pad))
        template_flipped = np.flip(template, axis=(0,1))
        template_hat = np.fft.fft2(template_flipped)
        return template_hat
    return (get_template_hat,)


@app.cell
def _(get_template_hat):
    template_hat = get_template_hat(1326)
    return (template_hat,)


@app.cell
def _(img_template_sum_projection_norm_2048_hat, np, plt, template_hat):
    template_matching = np.fft.fftshift(np.real(np.fft.ifft2(template_hat * img_template_sum_projection_norm_2048_hat)))
    plt.imshow(template_matching)
    return (template_matching,)


@app.cell
def _(np, template_matching):
    best_match = np.unravel_index(np.argmax(template_matching), template_matching.shape)
    np.max(template_matching)
    return (best_match,)


@app.cell
def _(best_match, np, plt, template_matching):
    plt.imshow(template_matching > np.max(template_matching)*0.999)
    plt.scatter(best_match[1], best_match[0])
    return


@app.cell
def _(best_match, img_template_sum_projection_norm_2048, plt):
    plt.imshow(img_template_sum_projection_norm_2048)
    plt.scatter(best_match[1], best_match[0])
    return


@app.cell
def _(plt, template):
    plt.imshow(template)
    return


@app.cell
def _(np):
    def red_green(red, green):
        img = np.zeros((*red.shape,3))
        img[:,:,0] = red
        img[:,:,1] = green
        return img
    return (red_green,)


@app.cell
def _(
    best_match,
    img_template_sum_projection_norm_2048,
    np,
    plt,
    red_green,
    template_1024,
):
    plt.imshow(red_green(np.roll(np.fft.fftshift(template_1024),(best_match[0],best_match[1]),axis=(0,1)) / 64 / 1,  img_template_sum_projection_norm_2048))
    return


@app.cell
def _(template_1024):
    template_1024.dtype
    return


@app.cell
def _(best_match):
    best_match
    return


@app.cell
def _(np, plt, template_1024):
    plt.imshow(np.flip(template_1024, axis=(0,1)))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
