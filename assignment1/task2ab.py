import matplotlib.pyplot as plt
import matplotlib
import pathlib
from utils import read_im, save_im
import numpy as np
matplotlib.use('Qt5Cairo')
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", ".\\lake.jpg"))
plt.imshow(im)


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
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")


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


plt.show()
