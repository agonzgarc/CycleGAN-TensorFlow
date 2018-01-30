import tensorflow as tf
import ops
import utils

class GeneratorDecoder:
  def __init__(self, name, is_training, nfs=32, nfe=16, norm='instance',
               image_size=32, reverse=False):
    self.name = name
    self.reuse = False
    self.nfs = nfs
    self.nfe = nfe
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.reverse = reverse
    self.conv_size = [32,16]

  def __call__(self, input):
    """
    Args:
      input: representation 8 x 8 x nf
    Returns:
      output: batch_size x width x height x 3
    """
    with tf.variable_scope(self.name):
        if self.reverse:
            g = tf.get_default_graph()
            with g.gradient_override_map({"Identity": "ReverseGrad"}):

                # gradient reversal layer
                idinput = tf.identity(input)

                #resnet blocks right after input
                res_input = ops.n_res_blocks(idinput, reuse=self.reuse, n=3)   # (?, w/4, h/4, nf)

                # fractional-strided convolutions
                fsconv1 = ops.uk(res_input, self.conv_size[0], is_training=self.is_training, norm=self.norm,
                      reuse=self.reuse, name='fsconv1')                 # (?, w/2, h/2, 8)
                fsconv2 = ops.uk(fsconv1, self.conv_size[1], is_training=self.is_training, norm=self.norm,
                      reuse=self.reuse, name='fsconv2', output_size=self.image_size)
                                                                        #(?, w, h, 4)
                # conv layer
                # Note: the paper said that ReLU and _norm were used
                # but actually tanh was used and no _norm here
                output = ops.c7s1_k(fsconv2, 3, norm=None,
                    activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)


        else:
            res_input = ops.n_res_blocks(input, reuse=self.reuse, n=3)   # (?, w/4, h/4, nf)

            #resnet blocks right after input
            # fractional-strided convolution
            fsconv1 = ops.uk(res_input, self.conv_size[0], is_training=self.is_training, norm=self.norm,
              reuse=self.reuse, name='fsconv1')                 # (?, w/2, h/2, 8)
            fsconv2 = ops.uk(fsconv1, self.conv_size[1], is_training=self.is_training, norm=self.norm,
              reuse=self.reuse, name='fsconv2', output_size=self.image_size)
                                                                #(?, w, h, 4)

            # conv layer
            # Note: the paper said that ReLU and _norm were used
            # but actually tanh was used and no _norm here
            output = ops.c7s1_k(fsconv2, 3, norm=None,
              activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)

        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

