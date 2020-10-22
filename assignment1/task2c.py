import numba as nb
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import numpy as np
from utils import read_im, save_im, normalize
matplotlib.use('Qt5Cairo')

output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)

im = read_im(pathlib.Path("images", "lake.jpg"))


def convolve_im(im, kernel,
                ):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    assert len(im.shape) == 3
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] == kernel.shape[0]
    dtype = im.dtype
    k_s = kernel.shape[0]
    k_s_div2 = kernel.shape[0] // 2
    image_padded = np.pad(im,
                          ((k_s_div2, k_s_div2),
                           (k_s_div2, k_s_div2),
                           (0, 0)))

    out = np.zeros(im.shape, im.dtype)
    # this it really really really slow, but shows how it's done
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            conv_sum = np.zeros(3)
            for k in range(k_s):
                for l in range(k_s):
                    for m in range(3):
                        conv_sum[m] += (image_padded[i:i+k_s, j:j+k_s, m][k, l]
                                        * kernel[-k-1, -l-1])

            out[i, j, :] = conv_sum[None, None, :]
    return out.astype(dtype)


# Define the convolutional kernels
h_b = 1 / 256 * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Convolve images
im_smoothed = convolve_im(im.copy(), h_b)
save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
im_sobel = convolve_im(im, sobel_x)
save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

# DO NOT CHANGE. Checking that your function returns as expected
assert isinstance(
    im_smoothed, np.ndarray),     f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
assert im_smoothed.shape == im.shape,     f"Expected smoothed im ({im_smoothed.shape}" + \
    f"to have same shape as im ({im.shape})"
assert im_sobel.shape == im.shape,     f"Expected smoothed im ({im_sobel.shape}" + \
    f"to have same shape as im ({im.shape})"

# %%
fig, ax = plt.subplots(1, 3)
ax[0].imshow(normalize(im))
ax[0].set_title('image')

ax[1].imshow(normalize(im_smoothed))
ax[1].set_title('image smoothed')

ax[2].imshow(normalize(im_sobel))
ax[2].set_title('image sobel')
fig.savefig(pathlib.Path('image_solutions', 'task2c.png'))

plt.show()
