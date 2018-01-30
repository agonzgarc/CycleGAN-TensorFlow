import tensorflow as tf
import ops
import utils

class GeneratorEncoderDisen:
  def __init__(self, name, is_training, nfs=32, nfe=16, norm='instance', image_size=32):
    self.name = name
    self.reuse = False
    self.nfs = nfs
    self.nfe = nfe
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.conv_size = [16,32]

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: split representation  8 x 8 x nfs+nfe
    """
    with tf.variable_scope(self.name):
        # conv layers
        conv1 = ops.c7s1_k(input, self.conv_size[0], is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='conv1')                           # (?,w, h, 4)
        conv2 = ops.dk(conv1, self.conv_size[1], is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='conv2')                               # (?,w/2, h/2, 8)
        # Fixed architecture, 2 chunks of ngf for shared, 1 for exclusive
        conv3 = ops.dk(conv2, self.nfs+self.nfe, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='conv3')                               # (?,w/4, h/4, 16)

        # use 3 residual blocks
        res_output = ops.n_res_blocks(conv3, reuse=self.reuse, n=3)     # (?,w/4, h/4, 16)

        shared = res_output[:,:,:,0:self.nfs]
        #print("size shared ",shared)
        exclusive = res_output[:,:,:,self.nfs:]

    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return shared, exclusive

