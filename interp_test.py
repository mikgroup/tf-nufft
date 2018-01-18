import numpy as np
import tensorflow as tf
import interp


class Test_util(tf.test.TestCase):

    def test_linear_interp(self):
        with self.test_session():
            table = [1.0, 0.0]
            x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            y = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
            self.assertAllClose(interp.linear_interp(table, x).eval(),
                                y)
