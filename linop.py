from itertools import product
import numpy as np
import tensorflow as tf
import logging
import util, interp


class Linop(object):
    '''
    Abstraction for linear operator
    A Linop must have an output and input shape.
    It can be added, subtracted, and composed / multiplied with another Linop or Bilinop.
    Its adjoint Linop can be obtained by .H.
    The user must specify both the _forward and _adjoint functions.

    Name scope can be added to the operations with the method add_name
    '''

    def __init__(self, oshape, ishape, dtype,
                 device=None, name=None):
        
        self.oshape = list(oshape)
        self.ishape = list(ishape)
        self.dtype = dtype
        self.name = self.__class__.__name__

        if device is None and tf.get_default_graph()._device_function_stack:
            self.device = tf.get_default_graph()._device_function_stack[0]
        else:
            self.device = device
        
        self.logger = logging.getLogger('{}.{}'.format(__name__, self.name))
        
    def check_domain(self, input):

        if input.dtype.base_dtype != self.dtype:
            raise ValueError('input data type mismatch, for {}, got {}'
                             .format(self, input.dtype))

        if util.get_shape(input) != self.ishape:
            raise ValueError('input shape mismatch, for {}, got {}'
                             .format(self, util.get_shape(input)))

    def check_codomain(self, output):

        if output.dtype.base_dtype != self.dtype:
            raise ValueError('output dtype mismatch, for {}, got {}'
                             .format(self, output.dtype))

        if util.get_shape(output) != self.oshape:
            raise ValueError('output shape mismatch, for {}, got {}'
                             .format(self, util.get_shape(output)))

    def _forward(self, input):
        raise NotImplementedError

    def _adjoint(self, input):
        raise NotImplementedError

    def forward(self, input):
        self.check_domain(input)

        with tf.device(self.device):
            output = self._forward(input)

        self.check_codomain(output)

        return output

    @property
    def H(self):
        return Adjoint(self)

    def __call__(self, input):
        return self.__mul__(input)

    def __mul__(self, input):
        if isinstance(input, Linop):
            return Compose(self, input)
        elif np.isscalar(input):
            return Scale(self, input)
        elif isinstance(input, (tf.Tensor, tf.SparseTensor, tf.Variable)):
            return self.forward(input)
        elif isinstance(input, np.ndarray):
            return self.forward(tf.convert_to_tensor(input))
        else:
            return NotImplemented

    def __rmul__(self, input):
        if np.isscalar(input):
            return Scale(self, input)
        else:
            return NotImplemented

    def __add__(self, input):
        if isinstance(input, Linop):
            return AddN([self, input])
        else:
            raise NotImplementedError

    def __neg__(self):
        return Scale(self, -1)

    def __sub__(self, input):
        return self.__add__(-input)

    def add_name_scope(self, name):
        
        self.name = name
            
        old_forward = self._forward
        old_adjoint = self._adjoint

        def _forward(input):
            with tf.name_scope(name):
                return old_forward(input)

        def _adjoint(input):
            with tf.name_scope(name + '.H'):
                return old_adjoint(input)

        self._forward = _forward
        self._adjoint = _adjoint

    def __repr__(self):
        return '<{}x{} {} Linop with dtype={}>'.format(self.oshape, self.ishape,
                                                       self.name, self.dtype)


class Adjoint(Linop):
    def __init__(self, linop):
        self._forward = linop._adjoint
        self._adjoint = linop._forward

        super().__init__(linop.ishape, linop.oshape, linop.dtype,
                         device=linop.device, name=linop.name + '.H')


class Scale(Linop):
    '''
    Scaled linear operator.

    Parameters
    ----------
    A : Linop.
    a : Scalar.

    Returns: a * A.
    '''

    def __init__(self, A, a):
        a = tf.cast(a, A.dtype)

        self._forward = lambda input: a * A._forward(input)
        self._adjoint = lambda input: tf.conj(a) * A._adjoint(input)

        super().__init__(A.oshape, A.ishape, A.dtype, name='Scaled' + A.name)

class AddN(Linop):
    ''' Returns linop1 + ... + linopN'''

    def __init__(self, linops, parallel=False):
        oshape = linops[0].oshape
        ishape = linops[0].ishape
        dtype = linops[0].dtype

        for linop in linops:
            assert linop.ishape == ishape
            assert linop.oshape == oshape
            assert linop.dtype == dtype

        super().__init__(oshape, ishape, dtype)
        self.linops = linops
        self.parallel = parallel

    def _forward(self, input):
        outputs = [self.linops[0]._forward(input)]
        for linop in self.linops[1:]:
            with tf.device(linop.device):
                if self.parallel:
                    outputs.append(linop._forward(input))
                else:
                    with tf.control_dependencies(outputs):
                        outputs.append(linop._forward(input))

        return sum(outputs)

    def _adjoint(self, input):

        outputs = [self.linops[0]._adjoint(input)]
        for linop in self.linops[1:]:
            with tf.device(linop.device):
                if self.parallel:
                    outputs.append(linop._adjoint(input))
                else:
                    with tf.control_dependencies(outputs):
                        outputs.append(linop._adjoint(input))

        return sum(outputs)


class Compose(Linop):
    '''Compose linop1 and linop2
    Performs x -> linop1(linop2(x))'''

    def __init__(self, linop1, linop2):

        if linop1.ishape != linop2.oshape:
            raise ValueError('Mismatched shapes for {0}, {1}'.
                             format(linop1, linop2))
        if linop1.dtype != linop2.dtype:
            raise ValueError('Mismatched dtypes for {0}, {1}'.
                             format(linop1, linop2))

        self.tshape = linop1.ishape

        super().__init__(linop1.oshape, linop2.ishape, linop1.dtype)

        self.linop1 = linop1
        self.linop2 = linop2

    def _forward(self, input):

        with tf.device(self.linop2.device):
            tmp = self.linop2._forward(input)

        with tf.control_dependencies([tmp]):
            with tf.device(self.linop1.device):
                return self.linop1._forward(tmp)

    def _adjoint(self, input):

        with tf.device(self.linop1.device):
            tmp = self.linop1._adjoint(input)

        with tf.control_dependencies([tmp]):
            with tf.device(self.linop2.device):
                return self.linop2._adjoint(tmp)


class Hstack(Linop):
    ''' Returns [linop1, ..., linopN]
    Takes a vector as input'''

    def __init__(self, linops, parallel=False):
        oshape = linops[0].oshape
        ishape = []
        dtype = linops[0].dtype

        for linop in linops:
            assert(linop.oshape == oshape)
            assert(linop.dtype == dtype)

        ishape = [sum([util.prod(linop.ishape)
                       for linop in linops])]

        super().__init__(oshape, ishape, dtype)
        self.linops = linops
        self.parallel = parallel

    def _forward(self, input):

        outputs = []

        for linop in self.linops:

            i = input[:util.prod(linop.ishape)]
            with tf.device(linop.device):
                if self.parallel:
                    outputs.append(linop._forward(tf.reshape(i, linop.ishape)))
                else:
                    with tf.control_dependencies(outputs):
                        outputs.append(linop._forward(
                            tf.reshape(i, linop.ishape)))

            input = input[util.prod(linop.ishape):]

        return sum(outputs)

    def _adjoint(self, input):

        outputs = []
        for linop in self.linops:

            with tf.device(linop.device):
                if self.parallel:
                    outputs.append(tf.reshape(linop._adjoint(input), [-1]))
                else:
                    with tf.control_dependencies(outputs):
                        outputs.append(tf.reshape(linop._adjoint(input), [-1]))

        return tf.concat(outputs, 0)


def Vstack(linops, parallel=False):
    return Hstack([linop.H for linop in linops], parallel=parallel).H


class Diag(Linop):
    '''
    Returns Diag([linop1, ..., linopN])
    Takes vectors as input and output
    '''

    def __init__(self, linops, parallel=False):
        oshape = []
        ishape = []

        ishape = [sum([util.prod(linop.ishape)
                       for linop in linops])]
        oshape = [sum([util.prod(linop.oshape)
                       for linop in linops])]

        dtype = linops[0].dtype
        for linop in linops:
            assert(linop.dtype == dtype)

        super().__init__(oshape, ishape, dtype)
        self.linops = linops
        self.parallel = parallel

    def _forward(self, input):
        outputs = []
        for linop in self.linops:
            i = tf.reshape(
                input[:util.prod(linop.ishape)], linop.ishape)
            with tf.device(linop.device):
                if self.parallel:
                    outputs.append(tf.reshape(linop._forward(i), [-1]))
                else:
                    with tf.control_dependencies(outputs):
                        outputs.append(tf.reshape(linop._forward(i), [-1]))

            input = input[util.prod(linop.ishape):]

        return tf.concat(outputs, 0)

    def _adjoint(self, input):
        outputs = []
        for linop in self.linops:
            i = tf.reshape(
                input[:util.prod(linop.oshape)], linop.oshape)
            with tf.device(linop.device):
                if self.parallel:
                    outputs.append(tf.reshape(linop._adjoint(i), [-1]))
                else:
                    with tf.control_dependencies(outputs):
                        outputs.append(tf.reshape(linop._adjoint(i), [-1]))
            input = input[util.prod(linop.oshape):]

        return tf.concat(outputs, 0)


class Identity(Linop):
    def __init__(self, shape, dtype=tf.complex64):
        super().__init__(shape, shape, dtype)

    def _forward(self, input):
        return input

    def _adjoint(self, input):
        return input


class Reshape(Linop):
    def __init__(self, oshape, ishape, dtype=tf.complex64):
        super().__init__(oshape, ishape, dtype)

    def _forward(self, input):
        return tf.reshape(input, self.oshape)

    def _adjoint(self, output):
        return tf.reshape(output, self.ishape)


class Transpose(Linop):

    def __init__(self, ishape, perm, dtype=tf.complex64):
        assert len(ishape) == len(perm)
        ndim = len(ishape)
        perm = [p % ndim for p in perm]
        oshape = [ishape[p] for p in perm]
        self.perm = perm
        self.iperm = np.argsort(perm)
        super().__init__(oshape, ishape, dtype)

    def _forward(self, input):
        return tf.transpose(input, self.perm)

    def _adjoint(self, input):
        return tf.transpose(input, self.iperm)


class MatMul(Linop):
    def __init__(self, oshape, ishape, mat, dtype=tf.complex64):
        self.mat = tf.convert_to_tensor(mat)

        mshape = util.get_shape(self.mat)
        max_ndim = max(len(oshape), len(ishape), len(mshape))

        oshape_full = [1] * (max_ndim - len(oshape)) + oshape
        ishape_full = [1] * (max_ndim - len(ishape)) + ishape
        mshape_full = [1] * (max_ndim - len(mshape)) + mshape

        # Check dimension valid.
        for i, o, m in zip(ishape_full[:-2], oshape_full[:-2], mshape_full[:-2]):

            if not ((i == m and o == m) or
                    (i == m and o == 1) or
                    (i == 1 and o == m) or
                    (i == 1 and o == 1) or
                    (i == o and m == 1)):
                raise ValueError('Invalid dimensions: {}, {}, {}'.
                                 format(oshape, ishape, mat.shape))

        self.osum_axes = [i for i in range(max_ndim)
                          if oshape_full[i] == 1]
        self.isum_axes = [i for i in range(max_ndim)
                          if ishape_full[i] == 1]
        ndim = len(oshape)
        self.perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]

        super().__init__(oshape, ishape, dtype)

    def _forward(self, input):
        output = tf.reduce_sum(self.mat @ input,
                               axis=self.osum_axes,
                               keep_dims=True)

        return tf.reshape(output, self.oshape)

    def _adjoint(self, input):
        output = tf.reduce_sum(tf.transpose(tf.conj(self.mat),
                                            perm=self.perm) @ input,
                               axis=self.isum_axes,
                               keep_dims=True)

        return tf.reshape(output, self.ishape)


class Sum(Linop):
    def __init__(self, ishape, axes, dtype=tf.complex64):
        oshape = [ishape[a] for a in range(len(ishape)) if a not in axes]

        self.tshape = ishape.copy()
        self.reps = [1] * len(ishape)
        for a in axes:
            self.tshape[a] = 1
            self.reps[a] = ishape[a]

        self.axes = axes

        super().__init__(oshape, ishape, dtype)

    def _forward(self, input):
        return tf.reduce_sum(input, axis=self.axes)

    def _adjoint(self, input):
        return tf.tile(tf.reshape(input, self.tshape), self.reps)


class Multiply(Linop):
    '''Returns a Linop that multiplies with mult
    and sums axes such that the output matches with oshape

    Parameters
    ----------
    oshape - output shape
    ishape - input shape
    mult - Tensor to be multiplied
    '''

    def __init__(self, oshape, ishape, mult, dtype=tf.complex64):

        self.mult = tf.convert_to_tensor(mult)
        if self.mult.dtype.base_dtype != dtype:
            self.mult = tf.cast(self.mult, dtype)
        super().__init__(oshape, ishape, dtype)

        mshape = util.get_shape(self.mult)
        max_ndim = max(max(len(oshape), len(ishape)), len(mshape))
        oshape = [1] * (max_ndim - len(oshape)) + list(oshape)
        ishape = [1] * (max_ndim - len(ishape)) + list(ishape)
        mshape = [1] * (max_ndim - len(mshape)) + list(mshape)

        self.osum_axis = [i for i in range(max_ndim) if (oshape[i] == 1 and
                                                         (ishape[i] > 1 or mshape[i] > 1))]
        self.isum_axis = [i for i in range(max_ndim) if (ishape[i] == 1 and
                                                         (oshape[i] > 1 or mshape[i] > 1))]

    def _forward(self, input):

        output = self.mult * input

        if self.osum_axis:
            output = tf.reduce_sum(output, axis=self.osum_axis)

        return tf.reshape(output, self.oshape)

    def _adjoint(self, input):

        output = tf.conj(self.mult) * input

        if self.isum_axis:
            output = tf.reduce_sum(output, axis=self.isum_axis)

        return tf.reshape(output, self.ishape)


class FFT(Linop):
    '''FFT linear operator

    Parameters
    ----------
    shape - input / output shape
    ndim - (optional) number of dimensions to apply FFT, defaults to None
           which does all dimensions. ndim must be 1, 2, or 3.
    '''

    def __init__(self, shape, ndim=None, center=True, dtype=tf.complex64):
        assert tf.complex64 == dtype
        self.ndim = ndim
        self.center = center
        super().__init__(shape, shape, dtype)

    def _forward(self, input):
        if self.center:
            return util.fftc(input, ndim=self.ndim)
        else:
            return util.fft(input, ndim=self.ndim)

    def _adjoint(self, input):
        if self.center:
            return util.ifftc(input, ndim=self.ndim)
        else:
            return util.ifft(input, ndim=self.ndim)


class Shift(Linop):

    def __init__(self, oshape, ishape, shift, dtype=tf.complex64):

        super().__init__(oshape, ishape, dtype)

        self.paddings = [[s, o - i - s]
                         for s, o, i in zip(shift, oshape, ishape)]
        self.begin = shift

    def _forward(self, input):

        real = tf.pad(tf.real(input), self.paddings, 'CONSTANT')

        with tf.control_dependencies([real]):

            imag = tf.pad(tf.imag(input), self.paddings, 'CONSTANT')

        return tf.complex(real, imag)

    def _adjoint(self, input):

        return tf.slice(input, self.begin, self.ishape)


class Interp(Linop):
    '''
    output - [batch, ...] or [..., batch]
    input - [batch, [nz], ny, nx] or [[nz], ny, nx, batch]
    '''

    def __init__(self, ishape, table, coord,
                 shift=None, dtype=tf.complex64):
        coord = tf.convert_to_tensor(coord)
        table = tf.convert_to_tensor(table)
        ndim = util.get_shape(coord)[-1]

        ishape = list(ishape)
        self.table = table
        self.coord = coord
        self.shift = shift
        batch = ishape[:-ndim]
        self.img_shape = ishape[-ndim:]
        oshape = batch + util.get_shape(coord)[:-1]

        super().__init__(oshape, ishape, dtype)

    def _forward(self, input):
        if self.shift is not None:
            coord = self.coord + self.shift
        else:
            coord = self.coord

        real = interp.interp(self.table, coord, tf.real(input))
        with tf.control_dependencies([real]):
            imag = interp.interp(self.table, coord, tf.imag(input))

        return tf.reshape(tf.complex(real, imag), self.oshape)

    def _adjoint(self, input):
        if self.shift is not None:
            coord = self.coord + self.shift
        else:
            coord = self.coord

        real = interp.interpH(self.ishape, self.table, coord, tf.real(input))
        with tf.control_dependencies([real]):
            imag = interp.interpH(self.ishape, self.table, coord, tf.imag(input))

        return tf.reshape(tf.complex(real, imag), self.ishape)


class Flip(Linop):
    def __init__(self, shape, axes, dtype=tf.complex64):
        self.axes = axes
        super().__init__(shape, shape, dtype)

    def _forward(self, input):
        return tf.reverse(input, axis=self.axes)

    def _adjoint(self, input):
        return tf.reverse(input, axis=self.axes)


class Zpad(Linop):
    '''Zero-pad linear operator
    Either centered or not centered, with optional shifted beginning

    Parameters
    ----------
    oshape - output shape
    ishape - input shape, must be smaller or equal to oshape
    center - (optional) centered or not boolean, defaults to True
    shift  - (optional) shift for beginning
    '''

    def __init__(self, oshape, ishape, center=True, shift=None, dtype=tf.complex64):

        super().__init__(oshape, ishape, dtype)
        self.center = center
        self.shift = shift

    def _forward(self, input):
        return util.zpad(input, self.oshape, shift=self.shift, center=self.center)

    def _adjoint(self, input):
        return util.crop(input, self.ishape, shift=self.shift, center=self.center)


def Crop(oshape, ishape, center=True, shift=None, dtype=tf.complex64):
    '''Crop linear operator
    Either centered or not centered

    Parameters
    ----------
    oshape - output shape
    ishape - input shape, must be greater or equal to oshape
    center - (optional) centered or not boolean, defaults to True
    shift  - (optional) shift for beginning
    '''

    A = Zpad(ishape, oshape, center=center, shift=shift, dtype=dtype).H

    return A


def sinh(x):

    return (tf.exp(x) + tf.exp(-x)) / 2


class KaiserApodize(Linop):
    '''
    Apodization with oversamp = 2, width = 2
    '''

    def __init__(self, shape, beta, ndim, dtype=tf.complex64):

        self.beta = beta
        self.ndim = ndim

        super().__init__(shape, shape, dtype)

    def _forward(self, input):
        output = input
        for d, i in zip(range(self.ndim), self.ishape[::-1]):
            with tf.control_dependencies([output]):
                idx = tf.reshape(tf.constant(
                    np.arange(i), dtype='float32'), [i] + [1] * d)
                x = (idx - i // 2) / i
                a = tf.sqrt(self.beta ** 2 - (np.pi * x) ** 2)
                output *= tf.cast(a / sinh(a), self.dtype)
        return output

    def _adjoint(self, input):
        output = input
        for d, i in zip(range(self.ndim), self.ishape[::-1]):
            with tf.control_dependencies([output]):
                idx = tf.reshape(tf.constant(
                    np.arange(i), dtype='float32'), [i] + [1] * d)
                x = (idx - i // 2) / i
                a = tf.sqrt(self.beta ** 2 - (np.pi * x) ** 2)
                output *= tf.cast(a / sinh(a), self.dtype)
        return output


class LinearPhase(Linop):
    def __init__(self, shape, phase, dtype=tf.complex64):
        self.phase = phase
        self.phase_conj = [-p for p in phase]
        super().__init__(shape, shape, dtype)

    def _forward(self, input):
        return util.linphase_mult(input, self.phase)

    def _adjoint(self, input):
        return util.linphase_mult(input, self.phase_conj)

    
def kb(x, width, beta):

    return 1 / width * np.i0(beta * np.sqrt(1 - x ** 2))


def NUFFT(ishape, coord, n=128, shifts=None, dtype=tf.complex64):
    '''
    ishape : [batch, nz, ny, nx] or [batch, ny, nx]
    coord : tensor of shape [..., ndim]
    width = 2.0
    oversamp = 2
    '''
    coord = tf.convert_to_tensor(coord)
    ishape = list(ishape)
    ndim = util.get_shape(coord)[-1]
    beta = np.pi * (1.5 ** 2 - 0.8) ** 0.5
    assert len(ishape) == ndim + 1

    # Apodization
    D = KaiserApodize(ishape, beta, ndim, dtype=dtype)

    # Get interpolation table
    kaiser_table = tf.constant(np.concatenate([kb(np.arange(n) / n, 2.0, beta), [0]]),
                               dtype='float32',
                               name='kaiser_table')

    # FFT
    F = FFT(ishape, ndim=ndim, dtype=dtype)

    As = []
    if shifts is None:
        shifts = list(product([0, 0.5], repeat=ndim))
    for shift in shifts:
        freq_shift = [s / i for s, i in zip(shift, ishape[-ndim:])]

        L = LinearPhase(ishape, freq_shift, dtype=dtype)
        I = Interp(ishape, kaiser_table, coord, shift=shift, dtype=dtype)

        As.append(I * F * L * D)

    A = AddN(As)
    A.add_name_scope('NUFFT')

    return A


class TensorToBlocks(Linop):

    def __init__(self, ishape, bshape, dtype=tf.complex64):

        ndim = len(bshape)

        for i, b in zip(ishape[-ndim:], bshape):
            assert(i % b == 0)

        batch_shape = ishape[:-ndim]
        nblocks = [i // b for i, b in zip(ishape[-ndim:], bshape)]
        oshape = batch_shape + nblocks + bshape
        super().__init__(oshape, ishape, dtype)

        self.ireshape = batch_shape.copy()
        for n, b in zip(nblocks, bshape):
            self.ireshape += [n, b]

        batch_ndim = len(batch_shape)
        self.iperm = (list(range(batch_ndim)) +
                      [batch_ndim + 2 * d for d in range(ndim)] +
                      [batch_ndim + 2 * d + 1 for d in range(ndim)])

        self.operm = list(range(batch_ndim))
        for i in range(ndim):
            self.operm += [batch_ndim + i, batch_ndim + ndim + i]

    def _forward(self, input):
        tmp = tf.reshape(input, self.ireshape)
        output = tf.transpose(tmp, perm=self.iperm)
        return tf.reshape(output, self.oshape)

    def _adjoint(self, input):
        return tf.reshape(tf.transpose(input, perm=self.operm), self.ishape)


def BlocksToTensor(oshape, bshape, dtype=tf.complex64):
    return TensorToBlocks(oshape, bshape, dtype=dtype).H


class Convolve(Linop):
    '''
    Convolution linear operator
    input  [batch_size, nx, ny, in_channels]
    filt   [kx, ky, in_channels, out_channels]
    output [batch_size, lx, ly, out_channels]

    mode = {'valid', 'full'}
    '''

    def __init__(self, ishape, filt, mode='full', dtype=tf.complex64):
        self.filt = tf.convert_to_tensor(filt)
        self.mode = mode
        self.ndim = len(ishape) - 2
        fshape = util.get_shape(self.filt)

        if mode == 'full':
            oshape = ([ishape[0]] +
                      [i1 + i2 - 1 for i1, i2 in zip(ishape[1:-1], fshape[:-2])]
                      + [fshape[-1]])
            self.mode_adj = 'valid'
        else:
            oshape = ([ishape[0]] +
                      [i1 - i2 + 1 for i1, i2 in zip(ishape[1:-1], fshape[:-2])]
                      + [fshape[-1]])
            self.zshape = ([ishape[0]] +
                           [i1 + i2 - 1 for i1, i2 in zip(ishape[1:-1], fshape[:-2])]
                           + [fshape[-1]])
            self.mode_adj = 'full'
        self.perm = list(range(self.ndim)) + [self.ndim + 1, self.ndim]
        self.filt_adj = tf.conj(tf.transpose(self.filt, self.perm))

        super().__init__(oshape, ishape, dtype)

    def _forward(self, input):
        return util.convolve(input, self.filt, mode=self.mode)

    def _adjoint(self, input):
        return util.correlate(input, self.filt_adj, mode=self.mode_adj)
