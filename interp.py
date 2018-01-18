import numpy as np
import tensorflow as tf
import util


def mod(x, n):

    with tf.name_scope('mod'):
        return x - n * (x // n)


def linear_interp(table, x):

    with tf.name_scope('linear_interp'):
        table = tf.convert_to_tensor(table)
        x = tf.convert_to_tensor(x)
        dtype = x.dtype.base_dtype
        
        n = util.get_size(table) - 1
        idx = tf.cast(x * n, tf.int64)

        frac = x * n - tf.cast(idx, dtype)
        left = tf.gather(table, tf.maximum(idx, 0))
        right = tf.gather(table, tf.minimum(idx + 1, n))

        return (1 - frac) * left + frac * right


def interp(table, coord, input):
    '''
    Interpolate onto coordinates with given interpolation table

    input - [num_channel, nx, ny]
    '''

    coord_shape = util.get_shape(coord)
    ishape = util.get_shape(input)
    ndim = coord_shape[-1]  # number of image dimensions
    bdim = len(ishape) - ndim  # number of channel dimensions
    kdim = len(coord_shape) - 1  # number of kspace dimensions

    img_shape = ishape[-ndim:]
    center = [i // 2 for i in img_shape]

    with tf.name_scope('get_indices'):
        idx = mod(tf.cast(tf.round(coord), 'int64') + center, img_shape)

    input = tf.transpose(input, perm=list(
        range(bdim, bdim + ndim)) + list(range(bdim)))

    output = tf.gather_nd(input, idx)

    output = tf.transpose(output, perm=list(
        range(kdim, kdim + bdim)) + list(range(kdim)))

    with tf.name_scope('get_weights'):
        diff = abs(tf.round(coord) - coord) * 2.0
        weight = tf.reduce_prod(linear_interp(table, diff), axis=-1)
        output *= weight

    return output


def interpH(oshape, table, coord, input):

    coord_shape = util.get_shape(coord)
    ndim = coord_shape[-1]
    bdim = len(oshape) - ndim
    kdim = len(coord_shape) - 1

    img_shape = oshape[-ndim:]
    center = [i // 2 for i in img_shape]

    idx = mod(tf.cast(tf.round(coord), 'int64') + center, img_shape)

    diff = abs(tf.round(coord) - coord) * 2.0
    weight = tf.reduce_prod(linear_interp(table, diff), axis=-1)
    input *= weight

    input = tf.transpose(input, perm=list(
        range(bdim, bdim + kdim)) + list(range(bdim)))
    output = tf.scatter_nd(idx, input, oshape[-ndim:] + oshape[:-ndim])

    output = tf.transpose(output, perm=list(
        range(ndim, ndim + bdim)) + list(range(ndim)))
        
    return output
