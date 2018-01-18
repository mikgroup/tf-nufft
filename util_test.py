import numpy as np
import tensorflow as tf
import util


def randn(shape, dtype='complex64'):
    if np.issubdtype(dtype, np.complex):
        real = np.random.normal(size=shape, scale=1 / (2 ** 0.5))
        imag = np.random.normal(size=shape, scale=1 / (2 ** 0.5))
        return (real + 1j * imag).astype(dtype)
    else:
        return np.random.normal(size=shape).astype(dtype)
    

class Test_util(tf.test.TestCase):

    def test_dirac(self):
        with self.test_session():
            output = util.dirac([5])
            truth = [0, 0, 1, 0, 0]
            self.assertAllClose(output.eval(), truth)

        with self.test_session():
            output = util.dirac([4])
            truth = [0, 0, 1, 0]
            self.assertAllClose(output.eval(), truth)

    def test_dot(self):
        for dtype in ['float32', 'complex64']:
            shape = [3, 4]
            x1 = randn(shape, dtype)
            x2 = randn(shape, dtype)

            with self.test_session():
                self.assertAllClose(np.vdot(x1, x2).real,
                                    util.dot(x1, x2).eval())

    def test_norm2(self):
        shape = [3, 4]
        x_ = randn(shape)
        x = tf.constant(x_)

        with self.test_session():
            self.assertAllClose(np.linalg.norm(x_.ravel())**2,
                                util.norm2(x).eval())

    def test_convolve(self):
        with self.test_session():
            l = np.array([1, 2, 3, 4], dtype='float32').reshape([1, 4, 1])
            l_ = tf.constant(l)
            r = np.array([0, 1, 0], dtype='float32').reshape([3, 1, 1])
            r_ = tf.constant(r)
            y_ = np.array([2, 3], dtype='float32').reshape([1, 2, 1])
            y = util.convolve(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1], dtype='float32').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([1, 2], dtype='float32').reshape([1, 2, 1])
            y = util.convolve(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

    def test_convolve_complex(self):
        with self.test_session():
            l = np.array([1j, 2j, 3j, 4j], dtype='complex64').reshape([1, 4, 1])
            l_ = tf.constant(l)
            r = np.array([0, 1j, 0], dtype='complex64').reshape([3, 1, 1])
            r_ = tf.constant(r)
            y_ = np.array([-2, -3], dtype='complex64').reshape([1, 2, 1])
            y = util.convolve(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1j], dtype='complex64').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([-1, -2], dtype='complex64').reshape([1, 2, 1])
            y = util.convolve(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

    def test_convolve_full(self):
        with self.test_session():
            l_ = np.array([1, 2, 3, 4], dtype='float32').reshape([1, 4, 1])
            l = tf.constant(l_)
            r_ = np.array([0, 1, 0], dtype='float32').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([0, 1, 2, 3, 4, 0], dtype='float32').reshape([1, 6, 1])
            y = util.convolve(l, r, mode='full')
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1], dtype='float32').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([0, 0, 1, 2, 3, 4], dtype='float32').reshape([1, 6, 1])
            y = util.convolve(l, r, mode='full')
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)
            
    def test_correlate(self):
        with self.test_session():
            l = np.array([1, 2, 3, 4], dtype='float32').reshape([1, 4, 1])
            l_ = tf.constant(l)
            r = np.array([0, 1, 0], dtype='float32').reshape([3, 1, 1])
            r_ = tf.constant(r)
            y_ = np.array([2, 3], dtype='float32').reshape([1, 2, 1])
            y = util.correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1], dtype='float32').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([3, 4], dtype='float32').reshape([1, 2, 1])
            y = util.correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

    def test_correlate_complex(self):
        with self.test_session():
            l = np.array([1j, 2j, 3j, 4j], dtype='complex64').reshape([1, 4, 1])
            l_ = tf.constant(l)
            r = np.array([0, 1j, 0], dtype='complex64').reshape([3, 1, 1])
            r_ = tf.constant(r)
            y_ = np.array([-2, -3], dtype='complex64').reshape([1, 2, 1])
            y = util.correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1j], dtype='complex64').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([-3, -4], dtype='complex64').reshape([1, 2, 1])
            y = util.correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

    def test_correlate_full(self):
        with self.test_session():
            l_ = np.array([1, 2, 3, 4], dtype='float32').reshape([1, 4, 1])
            l = tf.constant(l_)
            r_ = np.array([0, 1, 0], dtype='float32').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([0, 1, 2, 3, 4, 0], dtype='float32').reshape([1, 6, 1])
            y = util.correlate(l, r, mode='full')
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1], dtype='float32').reshape([3, 1, 1])
            r = tf.constant(r_)
            y_ = np.array([1, 2, 3, 4, 0, 0], dtype='float32').reshape([1, 6, 1])
            y = util.correlate(l, r, mode='full')
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)
             
    def test_outer_correlate(self):
        with self.test_session():
            l = np.array([1, 2, 3, 4], dtype='float32').reshape([1, 4, 1])
            l_ = tf.constant(l)
            r = np.array([0, 1, 0], dtype='float32').reshape([1, 3, 1])
            r_ = tf.constant(r)
            y_ = np.array([2, 3], dtype='float32').reshape([2, 1, 1])
            y = util.outer_correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1], dtype='float32').reshape([1, 3, 1])
            r = tf.constant(r_)
            y_ = np.array([3, 4], dtype='float32').reshape([2, 1, 1])
            y = util.outer_correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

    def test_outer_correlate_complex(self):
        with self.test_session():
            l = np.array([1j, 2j, 3j, 4j], dtype='complex64').reshape([1, 4, 1])
            l_ = tf.constant(l)
            r = np.array([0, 1j, 0], dtype='complex64').reshape([1, 3, 1])
            r_ = tf.constant(r)
            y_ = np.array([-2, -3], dtype='complex64').reshape([2, 1, 1])
            y = util.outer_correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

            r_ = np.array([0, 0, 1j], dtype='complex64').reshape([1, 3, 1])
            r = tf.constant(r_)
            y_ = np.array([-3, -4], dtype='complex64').reshape([2, 1, 1])
            y = util.outer_correlate(l, r)
            self.assertAllClose(y.eval(), y_, atol=1e-5, rtol=1e-5)

    def test_fft(self):
        x = randn([3, 4], 'complex64')
        y = np.fft.fftn(x, norm='ortho')
        y_ = util.fft(x)
        with self.test_session():
            self.assertAllClose(y_.eval(), y)

    def test_ifft(self):
        x = randn([3, 4], 'complex64')
        y = np.fft.ifftn(x, norm='ortho')
        y_ = util.ifft(x)
        with self.test_session():
            self.assertAllClose(y_.eval(), y)
