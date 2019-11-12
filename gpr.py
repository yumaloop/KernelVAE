"""
Gaussian process regression model based on GPflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("print_kernel", False, "Option to print out kernel")


class GaussianProcessRegression(object):
    """Gaussian process regression model based on GPflow.

  Args:
    input_x: numpy array, [data_size, input_dim]
    output_x: numpy array, [data_size, output_dim]
    kern: NNGPKernel class
  """

    def __init__(self, input_x, output_y, kern):
        with tf.name_scope("init"):
            self.input_x = input_x
            self.output_y = output_y
            self.num_train = input_x.shape[0]
            self.input_dim = input_x.shape[1]
            self.output_dim = output_y.shape[1]

            self.kern = kern # kernel function
            
            self.stability_eps = tf.identity(tf.placeholder(tf.float64))
            self.current_stability_eps = 1e-10

            # placeholder of output_y
            self.y_pl = tf.placeholder(
                tf.float64, [self.num_train, self.output_dim], name="y_train"
            )
            # placeholder of input_x
            self.x_pl = tf.identity(
                tf.placeholder(
                    tf.float64, [self.num_train, self.input_dim], name="x_train"
                )
            )

            self.l_np = None  # self.lの計算結果が入るPlaceholder
            self.v_np = None  # self.vの計算結果が入るPlaceholder
            self.k_np = None  # k_data_dataの計算結果が入るPlaceholder

        self.k_data_data = tf.identity(self.kern.k_full(self.x_pl))

    def _build_predict(self, n_test, full_cov=False):
        with tf.name_scope("build_predict"):
            self.x_test_pl = tf.identity(
                tf.placeholder(tf.float64, [n_test, self.input_dim], name="x_test_pl")
            )

        tf.logging.info("Using pre-computed Kernel")
        self.k_data_test = self.kern.k_full(self.x_pl, self.x_test_pl) # k_xt

        with tf.name_scope("build_predict"):
            a = tf.matrix_triangular_solve(self.l, self.k_data_test) # La = K_xt
            fmean = tf.matmul(a, self.v, transpose_a=True) # a^T v

            if full_cov:
                fvar = self.kern.k_full(self.x_test_pl) - tf.matmul(a, a, transpose_a=True)
                shape = [1, 1, self.y_pl.shape[1]]
                fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
            else:
                fvar = self.kern.k_diag(self.x_test_pl) - tf.reduce_sum(tf.square(a), 0)
                fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, self.output_y.shape[1]])

            self.fmean = fmean
            self.fvar = fvar

    def _build_cholesky(self):
        """
        self.l(L)とself.v(v)を計算する
        - L(L^T) = K_xx + eps I_n
        - Lv = Y
        """
        tf.logging.info("Computing Kernel")
        self.k_data_data_reg = (
            self.k_data_data
            + tf.eye(self.input_x.shape[0], dtype=tf.float64) * self.stability_eps
        )
        if FLAGS.print_kernel:
            self.k_data_data_reg = tf.Print(
                self.k_data_data_reg,
                [self.k_data_data_reg],
                message="K_DD = ",
                summarize=100,
            )
        self.l = tf.cholesky(self.k_data_data_reg) # K = L(L^T)
        self.v = tf.matrix_triangular_solve(self.l, self.y_pl) 
        # https://qiita.com/ToshihiroNakae/items/55d51e0f8ca48ae0d913#%E3%82%B3%E3%83%AC%E3%82%B9%E3%82%AD%E3%83%BC%E5%88%86%E8%A7%A3%E3%81%AB%E3%82%88%E3%82%8B%E7%AE%97%E5%87%BA

    def predict(self, test_x, sess, get_var=False):
        """Compute mean and varaince prediction for test inputs.
        Raises:
          ArithmeticError: Cholesky fails even after increasing to large values of stability epsilon.
        """
        if self.l_np is None:
            self._build_cholesky() # make and calc init L, v
            start_time = time.time()
            
            # calc K_DD by input_x
            self.k_np = sess.run(
                self.k_data_data, 
                feed_dict={self.x_pl: self.input_x}
            )
            tf.logging.info("Computed K_DD in %.3f secs" % (time.time() - start_time))
            
            # calc L, v 
            # 1回しかループしていない！
            while self.current_stability_eps < 1:
                try:
                    start_time = time.time()
                    # lとvを計算
                    self.l_np, self.v_np = sess.run(
                        [self.l, self.v],
                        feed_dict={
                            self.y_pl: self.output_y,
                            self.k_data_data: self.k_np,
                            self.stability_eps: self.current_stability_eps,
                        },
                    )
                    tf.logging.info(
                        "Computed L_DD in %.3f secs" % (time.time() - start_time)
                    )
                    break

                except tf.errors.InvalidArgumentError:
                    self.current_stability_eps *= 10
                    tf.logging.info(
                        "Cholesky decomposition failed, trying larger epsilon"
                        ": {}".format(self.current_stability_eps)
                    )

        if self.current_stability_eps > 0.2:
            raise ArithmeticError("Could not compute Cholesky decomposition.")

        n_test = test_x.shape[0]
        self._build_predict(n_test)
        feed_dict = {
            self.x_pl: self.input_x,
            self.x_test_pl: test_x,
            self.l: self.l_np,
            self.v: self.v_np,
        }

        start_time = time.time()
        if get_var:
            mean_pred, var_pred = sess.run([self.fmean, self.fvar], feed_dict=feed_dict)
            tf.logging.info("Did regression in %.3f secs" % (time.time() - start_time))
            return mean_pred, var_pred, self.current_stability_eps

        else:
            mean_pred = sess.run(self.fmean, feed_dict=feed_dict)
            tf.logging.info("Did regression in %.3f secs" % (time.time() - start_time))
            return mean_pred, self.current_stability_eps
