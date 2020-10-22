import numpy as np

img = np.array([[1, 0, 2, 3, 1], [3, 2, 0, 7, 0], [0, 6, 1, 1, 4]])
hist = np.bincount(img.ravel())

print(f'Histogram\n{np.vstack((np.arange(hist.size), hist)).T}')

cdf = np.cumsum(hist)
print(f'CDF\n{np.vstack((np.arange(cdf.size), cdf)).T}')


def h(v, cdf=cdf):
    return np.floor(np.amax(v)
                    * (cdf[v]-np.amin(cdf)).astype(float)
                    / (np.amax(cdf) - np.amin(cdf))).astype(int)


img_equalized = h(img)

print(f'Equalized \n{img_equalized}')
hist_equalized = np.bincount(img_equalized.ravel())
cdf_equalized = np.cumsum(hist_equalized)
print(f'CDF eqalized\n'
      f'{np.vstack((np.arange(cdf_equalized.size), cdf_equalized)).T}')
