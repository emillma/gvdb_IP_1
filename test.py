import numpy as np
import scipy.signal

import numpy as np
image = np.array([[1, 0, 2, 3, 1], [3, 2, 0, 7, 0], [0, 6, 1, 1, 4]])
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
out = scipy.signal.convolve2d(image, kernel, mode='same')
print(out)
