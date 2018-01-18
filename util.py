import os
import pywt
import random
import string
import tensorflow as tf
import numpy as np
from itertools import product
from tensorflow.python.client import timeline


def prod(x):
    return np.prod(x, dtype=np.int)


def vec(input):
    return tf.reshape(input, [-1])


def get_shape(input):
    return input.get_shape().as_list()


def get_ndim(input):
    return len(get_shape(input))


def get_size(input):
    return prod(get_shape(input))


def get_dtype(input):
    return input.dtype.base_dtype


def zpad(input, oshape, shift=None, center=True):
    input = tf.convert_to_tensor(input)
    ishape = get_shape(input)
    if shift is None:
        shift = [0] * len(ishape)

    if center:
        paddings = [[o // 2 + (-i + 1) // 2 + s, (o + 1) // 2 - (i + 1) // 2 - s]
                    for o, i, s in zip(oshape, ishape, shift)]
    else:
        paddings = [[0 + s, o - i - s]
                    for o, i, s in zip(oshape, ishape, shift)]

    with tf.name_scope('zpad'):
        if input.dtype.is_complex:
            real = tf.pad(tf.real(input), paddings, 'CONSTANT')
            with tf.control_dependencies([real]):
                imag = tf.pad(tf.imag(input), paddings, 'CONSTANT')

            return tf.complex(real, imag)
        else:
            return tf.pad(input, paddings, 'CONSTANT')


def crop(input, ishape, shift=None, center=True):
    input = tf.convert_to_tensor(input)
    oshape = get_shape(input)
    if shift is None:
        shift = [0] * len(oshape)

    if center:
        begin = [o // 2 + (-i + 1) // 2 + s for o, i,
                 s in zip(oshape, ishape, shift)]
    else:
        begin = shift

    with tf.name_scope('crop'):
        if input.dtype.is_complex:
            real = tf.slice(tf.real(input), begin, ishape)
            with tf.control_dependencies([real]):
                imag = tf.slice(tf.imag(input), begin, ishape)

            return tf.complex(real, imag)
        else:
            return tf.slice(input, begin, ishape)


def rss(x, axes=[0], keep_dims=False):
    with tf.name_scope('rss'):
        return tf.reduce_sum(abs(x)**2, axis=axes, keep_dims=keep_dims)**0.5


def dot(input1, input2):
    input1 = tf.convert_to_tensor(input1)
    input2 = tf.convert_to_tensor(input2)

    assert get_dtype(input1) == get_dtype(input2)
    dtype = get_dtype(input1)

    with tf.name_scope('dot'):
        if dtype.is_complex:
            real = tf.reduce_sum(tf.real(input1) * tf.real(input2))
            with tf.control_dependencies([real]):
                imag = tf.reduce_sum(tf.imag(input1) * tf.imag(input2))
            return real + imag
        else:
            return tf.reduce_sum(input1 * input2)


def norm(input, axes=None, keep_dims=False):
    with tf.name_scope('norm'):
        return norm2(input, axes=axes, keep_dims=keep_dims)**0.5


def norm2(input, axes=None, keep_dims=False):
    with tf.name_scope('norm2'):
        return tf.reduce_sum(abs(input)**2, axis=axes, keep_dims=keep_dims)


def check_conv_inputs(input1, input2):
    
    assert input1.dtype.base_dtype == input2.dtype.base_dtype
    dtype = input1.dtype.base_dtype
    ishape1 = get_shape(input1)
    ishape2 = get_shape(input2)
        
    assert len(ishape1) == len(ishape2)
    ndim = len(ishape2) - 2
    assert ndim == 1 or ndim == 2 or ndim == 3
    assert (dtype == 'float16' or dtype == 'float32' or
            dtype == 'complex32' or dtype == 'complex64')
    assert ishape1[-1] == ishape2[-2]
    return ishape1, ishape2, dtype


def convolve(input1, input2, mode='valid'):
    '''
    Supports full and valid mode, and complex dtype.

    conv2/3d_transpose with mode='VALID' actually implements convolution with mode='full'
    For example for 2D,
    input1 of [batch_size, nx, ny, in_channels]
    input2 of [kx, ky, in_channels, out_channels]
    '''
    with tf.name_scope('convolve'):
        input1 = tf.convert_to_tensor(input1)
        input2 = tf.convert_to_tensor(input2)
        ishape1, ishape2, dtype = check_conv_inputs(input1, input2)

        ndim = len(ishape1) - 2
        if ndim == 1:
            output = convolve(tf.expand_dims(input1, axis=1),
                              tf.expand_dims(input2, axis=0),
                              mode=mode)
            return tf.squeeze(output, axis=1)
        elif ndim == 2:
            conv = tf.nn.conv2d_transpose
        elif ndim == 3:
            conv = tf.nn.conv3d_transpose
        else:
            raise ValueError('ndim can only be 1, 2, or 3')
        
        full_oshape = ([ishape1[0]] + [i1 + i2 - 1
                                       for i1, i2 in zip(ishape1[1:-1], ishape2[:-2])]
                       + [ishape2[-1]])
        perm = list(range(ndim)) + [ndim + 1, ndim]
        input2 = tf.transpose(input2, perm)

        padding = 'VALID'
        strides = [1] * (ndim + 2)
        if dtype.is_complex:
            realreal = conv(tf.real(input1), tf.real(input2), full_oshape, strides, padding)
            with tf.control_dependencies([realreal]):
                imagimag = conv(tf.imag(input1), tf.imag(input2), full_oshape, strides, padding)
            real = realreal - imagimag

            with tf.control_dependencies([real]):
                realimag = conv(tf.real(input1), tf.imag(input2), full_oshape, strides, padding)
            with tf.control_dependencies([realimag]):
                imagreal = conv(tf.imag(input1), tf.real(input2), full_oshape, strides, padding)
            imag = realimag + imagreal
            
            full_output = tf.complex(real, imag)
        else:
            full_output = conv(input1, input2, full_oshape, strides, padding)

        if mode == 'full':
            return full_output
        else:
            valid_oshape = ([ishape1[0]] + [i1 - i2 + 1
                                            for i1, i2 in zip(ishape1[1:-1], ishape2[:-2])]
                            + [ishape2[-1]])
            return crop(full_output, valid_oshape)
            

def correlate(input1, input2, mode='valid'):
    '''
    Supports full and valid mode, and complex dtype.

    conv2/3d with mode 'VALID' actually implements correlation with mode='valid'
    For example for 2D,
    input1 of [batch_size, nx, ny, in_channels]
    input2 of [kx, ky, in_channels, out_channels]
    '''
    with tf.name_scope('correlate'):
        input1 = tf.convert_to_tensor(input1)
        input2 = tf.convert_to_tensor(input2)
        ishape1, ishape2, dtype = check_conv_inputs(input1, input2)
        
        ndim = len(ishape1) - 2
        if ndim == 1:
            output = correlate(tf.expand_dims(input1, axis=1),
                               tf.expand_dims(input2, axis=0),
                               mode=mode)
            return tf.squeeze(output, axis=1)
        elif ndim == 2:
            corr = tf.nn.conv2d
        elif ndim == 3:
            corr = tf.nn.conv3d
        else:
            raise ValueError('ndim can only be 1, 2, or 3')

        if mode == 'full':
            full_ishape1 = ([ishape1[0]] + [i1 + 2 * i2 - 2
                                       for i1, i2 in zip(ishape1[1:-1], ishape2[:-2])]
                            + [ishape1[-1]])
            input1 = zpad(input1, full_ishape1)

        padding = 'VALID'
        strides = [1] * (ndim + 2)
        if dtype.is_complex:
            realreal = corr(tf.real(input1), tf.real(input2), strides, padding)
            with tf.control_dependencies([realreal]):
                imagimag = corr(tf.imag(input1), tf.imag(input2), strides, padding)
            real = realreal - imagimag

            with tf.control_dependencies([real]):
                realimag = corr(tf.real(input1), tf.imag(input2), strides, padding)
            with tf.control_dependencies([realimag]):
                imagreal = corr(tf.imag(input1), tf.real(input2), strides, padding)
            imag = realimag + imagreal
            
            return tf.complex(real, imag)
        else:
            return corr(input1, input2, strides, padding)

        
def check_outer_corr_inputs(input1, input2):
    
    assert input1.dtype.base_dtype == input2.dtype.base_dtype
    dtype = input1.dtype.base_dtype
    ishape1 = get_shape(input1)
    ishape2 = get_shape(input2)
        
    assert len(ishape1) == len(ishape2)
    ndim = len(ishape2) - 2
    assert ndim == 1 or ndim == 2 or ndim == 3
    assert (dtype == 'float16' or dtype == 'float32' or
            dtype == 'complex32' or dtype == 'complex64')
    assert ishape1[0] == ishape2[0]
    return ishape1, ishape2, dtype


def outer_correlate(input1, input2):
    '''
    Supports complex dtype.
    ishape1 >= ishape2

    conv2/3d_backprop implements correlation with mode='valid'
    For example for 2D,
    input1 of [batch_size, nx, ny, channel1]
    input2 of [batch_size, lx, ly, channel2]
    output of [kx, ky, channel1, channel2]
    where kx = nx - lx + 1
    '''
    with tf.name_scope('correlate'):
        input1 = tf.convert_to_tensor(input1)
        input2 = tf.convert_to_tensor(input2)
        ishape1, ishape2, dtype = check_outer_corr_inputs(input1, input2)
        
        ndim = len(ishape1) - 2
        if ndim == 1:
            output = outer_correlate(tf.expand_dims(input1, axis=1),
                                     tf.expand_dims(input2, axis=1))
            return tf.squeeze(output, axis=0)
        elif ndim == 2:
            corr = tf.nn.conv2d_backprop_filter
        elif ndim == 3:
            corr = tf.nn.conv3d_backprop_filter_v2
        else:
            raise ValueError('ndim can only be 1, 2, or 3')
        
        oshape = ([i1 - i2 + 1 for i1, i2 in zip(ishape1[1:-1], ishape2[1:-1])] +
                  [ishape1[-1], ishape2[-1]])

        padding = 'VALID'
        strides = [1] * (ndim + 2)
        if dtype.is_complex:
            realreal = corr(tf.real(input1), oshape, tf.real(input2), strides, padding)
            with tf.control_dependencies([realreal]):
                imagimag = corr(tf.imag(input1), oshape, tf.imag(input2), strides, padding)
            real = realreal - imagimag

            with tf.control_dependencies([real]):
                realimag = corr(tf.real(input1), oshape, tf.imag(input2), strides, padding)
            with tf.control_dependencies([realimag]):
                imagreal = corr(tf.imag(input1), oshape, tf.real(input2), strides, padding)
            imag = realimag + imagreal
            
            return tf.complex(real, imag)
        else:
            return corr(input1, oshape, input2, strides, padding)


def vec(inputs):
    with tf.name_scope('vec'):
        return tf.concat([tf.reshape(i, [-1]) for i in inputs], 0)


def split(vec, oshapes):
    with tf.name_scope('split'):
        outputs = []
        for oshape in oshapes:
            osize = np.prod(oshape, dtype=np.int)
            outputs.append(tf.reshape(vec[:osize], oshape))
            vec = vec[osize:]

        return outputs


def dirac(shape, dtype=tf.complex64):
    with tf.name_scope('dirac'):
        ndim = len(shape)
        idx = shape[0] // 2
        for i in range(1, ndim):
            idx = shape[i - 1] * idx + shape[i] // 2

        size = np.prod(shape, dtype=np.int)
        return tf.reshape(tf.one_hot(idx, size, dtype=dtype), shape)


def soft_thresh(t, x):
    with tf.name_scope('soft_thresh'):
        x = tf.convert_to_tensor(x)
        dtype = x.dtype.base_dtype

        mag = abs(x) - tf.cast(t, dtype.real_dtype)
        mag = tf.cast((abs(mag) + mag) / 2, dtype)

        tol = 1e-5
        return mag * (x / tf.cast(abs(x) + tol, dtype))


def elitist_thresh(lamda, x, axes=None):
    with tf.name_scope('elitist_thresh'):
        x = tf.convert_to_tensor(x)
        if axes is not None:
            batch_axes = [i for i in range(get_ndim(x)) if i not in axes]
            perm = batch_axes + axes
            x = tf.transpose(x, perm)
            shape = get_shape(x)
            ndim = len(axes)
        else:
            shape = get_shape(x)
            ndim = len(shape)

        length = prod(shape[-ndim:])
        batch = prod(shape[:-ndim])
        x = tf.reshape(x, [-1, length])

        s = tf.nn.top_k(abs(x), k=length).values
        l1 = tf.cumsum(s, axis=-1)
        i = tf.constant(np.arange(length), dtype=l1.dtype)
        lamda = tf.cast(lamda, l1.dtype)
        ts = l1 * lamda / (1 + lamda * (i + 1))
        idx = tf.stack([tf.range(batch, dtype=tf.int64),
                        tf.argmax(tf.cast(ts[:, :-1] > s[:, 1:], tf.float32), axis=-1)])
        idx = tf.transpose(idx)
        t = tf.expand_dims(tf.gather_nd(ts, idx), axis=1)

        y = soft_thresh(t, x)
        y = tf.reshape(y, shape)
        
        if axes is not None:
            y = tf.transpose(y, np.argsort(perm))

        return y


def rand(shape, dtype=tf.complex64):
    with tf.name_scope('rand'):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype.is_complex:
            r = randn(shape, dtype=dtype)
            return r / tf.cast(abs(r), dtype)
        else:
            return tf.random_uniform(shape, dtype=dtype, minval=-1, maxval=1)


def randn(shape, dtype=tf.complex64):
    with tf.name_scope('randn'):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype.is_complex:
            return tf.complex(tf.random_normal(shape, dtype=dtype.real_dtype),
                              tf.random_normal(shape, dtype=dtype.real_dtype))
        else:
            return tf.random_normal(shape, dtype=dtype)


def fft(input, ndim=None, dtype=tf.complex64):
    with tf.name_scope('fft'):
        assert tf.complex64 == dtype
        if not isinstance(input, tf.Variable):
            input = tf.convert_to_tensor(input)
        shape = get_shape(input)

        if ndim is None:
            ndim = len(shape)
        if ndim == 1:
            fft = tf.fft
        elif ndim == 2:
            fft = tf.fft2d
        elif ndim == 3:
            fft = tf.fft3d
        else:
            raise ValueError('ndim must be 1, 2 or 3')

        scale = np.prod(shape[-ndim:], dtype=np.int)**0.5
        return fft(input) / scale


def ifft(input, ndim=None):
    with tf.name_scope('ifft'):
        if not isinstance(input, tf.Variable):
            input = tf.convert_to_tensor(input)
        assert tf.complex64 == input.dtype.base_dtype
        shape = get_shape(input)

        if ndim is None:
            ndim = len(shape)
        if ndim == 1:
            ifft = tf.ifft
        elif ndim == 2:
            ifft = tf.ifft2d
        elif ndim == 3:
            ifft = tf.ifft3d
        else:
            raise ValueError('ndim must be 1, 2 or 3')

        scale = np.prod(shape[-ndim:], dtype=np.int)**0.5
        return ifft(input) * scale


def fftmod(input, ndim=None, dtype=tf.complex64):
    if not isinstance(input, tf.Variable):
        input = tf.convert_to_tensor(input)
    assert tf.complex64 == input.dtype.base_dtype
    shape = get_shape(input)
    if ndim is None:
        ndim = len(shape)

    output = input
    for d, i in zip(range(ndim), shape[::-1]):
        with tf.control_dependencies([output]):
            idx = tf.reshape(tf.constant(
                np.arange(i), dtype='float32'), [i] + [1] * d)
            arg = (idx - (i // 2) / 2) * ((i // 2) / i)
            exp = tf.complex(tf.cos(2 * np.pi * arg), tf.sin(2 * np.pi * arg))
            output *= exp
    return output


def ifftmod(input, ndim=None, dtype=tf.complex64):
    if not isinstance(input, tf.Variable):
        input = tf.convert_to_tensor(input)
    assert tf.complex64 == input.dtype.base_dtype
    shape = get_shape(input)
    if ndim is None:
        ndim = len(shape)

    output = input
    for d, i in zip(range(ndim), shape[::-1]):
        with tf.control_dependencies([output]):
            idx = tf.reshape(tf.constant(
                np.arange(i), dtype='float32'), [i] + [1] * d)
            arg = -(idx - (i // 2) / 2) * ((i // 2) / i)
            exp = tf.complex(tf.cos(2 * np.pi * arg), tf.sin(2 * np.pi * arg))
            output *= exp
    return output


def fftc(input, ndim=None):
    with tf.name_scope('fftc'):
        return fftmod(fft(fftmod(input, ndim), ndim), ndim)


def ifftc(input, ndim=None):
    with tf.name_scope('ifftc'):
        return ifftmod(ifft(ifftmod(input, ndim), ndim), ndim)


class TraceSession(object):

    def __init__(self, sess):
        self.sess = sess
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                                     output_partition_graphs=True)
        self.metadata_dict = {}

    def _valid_filename(self, filename):
        filename = filename.replace('/', '_')
        valid_chars = ", -_%s%s" % (string.ascii_letters, string.digits)
        return ''.join(c for c in filename if c in valid_chars)[:200]

    def run(self, ops):
        if isinstance(ops, list):
            ops_str = ', '.join(o.name for o in ops)
        else:
            ops_str = ops.name
        ops_str = self._valid_filename(ops_str)
        if ops_str not in self.metadata_dict:
            self.metadata_dict[ops_str] = tf.RunMetadata()
            
        return self.sess.run(ops, options=self.options,
                             run_metadata=self.metadata_dict[ops_str])

    def write_timeline(self, folder):
        
        if not os.path.exists(folder):
            os.makedirs(folder)

        for name, metadata in self.metadata_dict.items():
            with open(os.path.join(folder, name + '.json'), 'w') as f:
                f.write(timeline.Timeline(metadata.step_stats).
                        generate_chrome_trace_format(show_memory=True,
                                                     show_dataflow=True))


def write_graph(folder):
    tf.summary.FileWriter(folder, graph=tf.get_default_graph())


def linphase_mult(input, phase):
    input = tf.convert_to_tensor(input)
    ishape = get_shape(input)
    ndim = len(phase)
    output = input
    for d, i, p in zip(range(ndim), ishape[::-1], phase[::-1]):
        with tf.control_dependencies([output]):
            idx = tf.reshape(tf.constant(np.arange(i), dtype='float32'), [i] + [1] * d)
            arg = (idx - i // 2) * p
            exp = tf.complex(tf.cos(2 * np.pi * arg),
                             tf.sin(2 * np.pi * arg))
            output *= exp

    return output
