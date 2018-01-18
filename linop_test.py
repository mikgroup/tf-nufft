import numpy as np
import tensorflow as tf
import linop, util


def randn(shape, dtype='complex64'):
    if np.issubdtype(dtype, np.complex):
        real = np.random.normal(size=shape, scale=1 / (2 ** 0.5))
        imag = np.random.normal(size=shape, scale=1 / (2 ** 0.5))
        return (real + 1j * imag).astype(dtype)
    else:
        return np.random.normal(size=shape).astype(dtype)


class TestLinop(tf.test.TestCase):

    def check_linop_adjoint(self, A):

        with tf.name_scope('check_linop_adjoint'):
            x = randn(A.ishape, A.dtype.as_numpy_dtype)
            y = randn(A.oshape, A.dtype.as_numpy_dtype)

            self.assertAllClose(util.dot(A(x), y).eval(),
                                util.dot(x, A.H(y)).eval(), atol=1e-5, rtol=1e-5)

    def test_MatMul(self):
        oshape = [5, 4, 3]
        ishape = [5, 2, 3]
        mshape = [5, 4, 2]
        A = linop.MatMul(oshape, ishape, randn(mshape, dtype='complex64'))
        with self.test_session():
            self.check_linop_adjoint(A)

    def test_FFT(self):

        with self.test_session():

            for shape in [[3, ], [4, 3], [5, 4, 3]]:
                for ndim in range(1, len(shape) + 1):

                    A = linop.FFT(shape, ndim=ndim)
                    self.check_linop_adjoint(A)

                    x_ = randn(shape, 'complex64')
                    x = tf.constant(x_)

                    y_ = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x_),
                                                     axes=range(-1, -ndim - 1, -1),
                                                     norm='ortho'))
                    y = A(x)

                    self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

    def test_Multiply(self):

        with self.test_session():

            mshape = [4, 3]
            ishape = [3]
            oshape = [3]

            with self.test_session():
                mult_ = randn(mshape, 'complex64')
                mult = tf.constant(mult_)

                A = linop.Multiply(oshape, ishape, mult)
                self.check_linop_adjoint(A)

                x_ = randn(ishape, 'complex64')
                x = tf.constant(x_)

                y_ = np.sum(mult_ * x_, axis=0)
                y = A(x)

            self.assertAllClose(y.eval(), y_)

    def test_KaiserApodize(self):

        ndim = 2
        ishape = [3, 2]
        beta = np.pi * (1.5 ** 2 - 0.8) ** 0.5
        A = linop.KaiserApodize(ishape, beta, ndim)
        
        with self.test_session():
            self.check_linop_adjoint(A)
        

    def test_NUFFT(self):

        with self.test_session():

            # Check deltas
            ishape = [1, 3]
            coord = tf.constant([[-1], [0], [1]], 'float32')
            A = linop.NUFFT(ishape, coord)
            self.check_linop_adjoint(A)

            x = np.array([[0, 1, 0]], 'complex64')  # delta
            self.assertAllClose(np.array([[1.0, 1.0, 1.0]], 'complex64') / (3 ** 0.5),
                                A(x).eval(),
                                atol=0.1, rtol=0.1)
            

    def test_Interp(self):

        with self.test_session():

            ishape = [1, 3, 4]
            table = tf.constant(np.random.random(10), dtype='float32')
            coord = tf.constant(np.random.random(
                [10, 2]) - 0.5, dtype='float32')

            A = linop.Interp(ishape, table, coord)
            self.check_linop_adjoint(A)

    def test_TensorToBlocks(self):

        ishape = [4]
        bshape = [2]

        with self.test_session():

            A = linop.TensorToBlocks(ishape, bshape)
            self.check_linop_adjoint(A)

            x_ = np.array([1, 2, 3, 4], 'complex64')
            x = tf.constant(x_)

            y_ = np.array([[1, 2],
                           [3, 4]], 'complex64')
            y = A(x)

            self.assertAllClose(y.eval(), y_)

    def test_Convolve(self):

        with self.test_session():
            ishape = [2, 5, 6, 3]
            filt = randn([2, 2, 3, 4], 'complex64')

            A = linop.Convolve(ishape, filt, mode='full')
            self.check_linop_adjoint(A)

            A = linop.Convolve(ishape, filt, mode='valid')
            self.check_linop_adjoint(A)
