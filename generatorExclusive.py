import tensorflow as tf
import ops
import utils

class GeneratorExclusive:
  def __init__(self, name, is_training, nfe=16, norm='instance',
               image_size=32, reverse=False):
    self.name = name
    self.reuse = False
    self.nfe = nfe
    self.norm = norm
    self.is_training = is_training
    self.reverse = reverse


  def __call__(self, input):
    """
    Args:
      input: batch_size x 1 x 1 x 10
    Returns:
      output: 8 x 8 x nfe
    """
    with tf.variable_scope(self.name):

        # fractional-strided convolution
        fsconv1 = ops.uk(input, 64, is_training=self.is_training, norm=self.norm,
            reuse=self.reuse, name='fsconv1', output_size=2)                 # (?, 2, 2, 64)

        fsconv2 = ops.uk(fsconv1, 32, is_training=self.is_training, norm=self.norm,
              reuse=self.reuse, name='fsconv2', output_size=4)
                                                                #(?, 4, 4, 32)

        fsconv3 = ops.uk(fsconv2, 16, is_training=self.is_training, norm=self.norm,
              reuse=self.reuse, name='fsconv3', output_size=8)
                                                                #(?, 8, 8, 16)

        # use 3 residual blocks
        res_output = ops.n_res_blocks(fsconv3, reuse=self.reuse, n=3)



    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return res_output

