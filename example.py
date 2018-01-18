import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from linop import NUFFT
from util import rss

# Get data
img = np.load('img.npy')
mps = np.load('mps.npy')
coord = np.load('coord.npy')

# Create NUFFT linear operator
# coord should be of type float32,
# and is scaled like BART (goes from -FOV / 2 to FOV / 2
# instead of -0.5 to 0.5)
F = NUFFT(mps.shape, coord, dtype=img.dtype)

# NUFFT Forward
ksp_tf = F(mps * img)
with tf.Session() as sess:
    ksp = ksp_tf.eval()

# NUFFT Adjoint
img_adj_tf = tf.reduce_sum(tf.conj(mps) * F.H(ksp), axis=0)
with tf.Session() as sess:
    img_adj = img_adj_tf.eval()

plt.figure()
plt.imshow(abs(img_adj), cmap='gray')
plt.show()
