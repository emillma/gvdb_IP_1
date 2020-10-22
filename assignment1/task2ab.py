import matplotlib.pyplot as plt
import matplotlib
import pathlib
from utils import read_im, save_im
import numpy as np


output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", ".\\lake.jpg"))


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    dtype = im.dtype
    color_values = np.array([0.212, 0.7152, 0.0722])
    im = np.sum(im * color_values[None, None, :], axis=-1)
    return im.astype(dtype)


im_greyscale = greyscale(im)
fig0, ax0 = plt.subplots(1, 2)
ax0[0].imshow(im)
ax0[0].set_title('color image')
ax0[1].imshow(im_greyscale, cmap="gray")
ax0[1].set_title('grayscale image')
fig0.tight_layout()
fig0.savefig(pathlib.Path('image_solutions', 'task2a.png'))

save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # YOUR CODE HERE
    dtype = im.dtype
    im = 1 - im
    return im.astype(dtype)


im_inv = inverse(im)
im_greyscale_inv = inverse(im_greyscale)

fig1, ax1 = plt.subplots(2, 2)
ax1[0, 0].imshow(im)
ax1[0, 0].set_title('color image')
ax1[0, 1].imshow(im_inv)
ax1[0, 1].set_title('color image inverted')
ax1[1, 0].imshow(im_greyscale, cmap="gray")
ax1[1, 0].set_title('grayscale image')
ax1[1, 1].imshow(im_greyscale_inv, cmap="gray")
ax1[1, 1].set_title('grayscale image inverted')
fig1.tight_layout()
fig1.savefig(pathlib.Path('image_solutions', 'task2b.png'))

plt.show()
