import tensorflow as tf
import ops
import utils

class GeneratorTransformer:
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
      input: batch_size x width x height x 16
    Returns:
      output: batch_size x width x height x 32 
    """
    with tf.variable_scope(self.name):
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": "ReverseGrad"}):
          #idinput = tf.identity(input)
          idinput = input
          res_input = ops.n_res_blocks(idinput, reuse=self.reuse, n=3)

          # conv layer  - output depth is 32 (as in shared representation)
          output = ops.c7s1_k(res_input, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          activation='tanh',reuse=self.reuse, name='c7s1_32') # (?, w, h, 32)

    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image
