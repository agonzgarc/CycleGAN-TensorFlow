import tensorflow as tf
import ops
import utils

class GeneratorDecoder:
  def __init__(self, name, is_training, ngf=16, norm='instance', image_size=32):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):

      #resnet blocks right after input
      res_input = ops.n_res_blocks(input, reuse=self.reuse, n=3)      # (?, w/4, h/4, 64)
      # fractional-strided convolution
      u32 = ops.uk(res_input, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32')                                 # (?, w/2, h/2, 32)
      u16 = ops.uk(u32, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u16', output_size=self.image_size)         #(?, w, h, 16)

      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here
      output = ops.c7s1_k(u16, 3, norm=None,
          activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)

      ##resnet blocks right after input
      #res_input = ops.n_res_blocks(input, reuse=self.reuse, n=3)      # (?, w/4, h/4, 128)
      ## fractional-strided convolution
      #u64 = ops.uk(res_input, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          #reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
      #u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
          #reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)

      ## conv layer
      ## Note: the paper said that ReLU and _norm were used
      ## but actually tanh was used and no _norm here
      #output = ops.c7s1_k(u32, 3, norm=None,
      #    activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)


    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

