import tensorflow as tf
import ops
import utils

class GeneratorEncoderDisen:
  #def __init__(self, name, is_training, num_shared=128, num_exclusive=64, ngf=64, norm='instance', image_size=128):
  def __init__(self, name, is_training, ngf=16, norm='instance', image_size=128):
    self.name = name
    #self.shared = num_shared
    #self.exclusive = num_exclusive
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
        # conv layers
        c7s1_16 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_16')                             # (?, w, h, 16)
        d32 = ops.dk(c7s1_16, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d32')                                 # (?, w/2, h/2, 32)
        # Fixed architecture, 2 chunks of ngf for shared, 1 for exclusive
        d48 = ops.dk(d32, 3*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d48')                                # (?, w/4, h/4, 64)

        # use 3 residual blocks
        res_output = ops.n_res_blocks(d48, reuse=self.reuse, n=3)      # (?, w/4, h/4, 64)

        shared = res_output[:,:,:,0:32]
        #print("size shared ",shared)
        exclusive = res_output[:,:,:,32:]

        ## conv layers
        #c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          #reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
        #d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          #reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
        ## Fixed architecture, 2 chunks of ngf for shared, 1 for exclusive
        #d128 = ops.dk(d64, 3*self.ngf, is_training=self.is_training, norm=self.norm,
          #reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

        ## use 3 residual blocks
        #res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=3)      # (?, w/4, h/4, 128)

        #shared = res_output[:,:,:,0:128]
        ##print("size shared ",shared)
        #exclusive = res_output[:,:,:,128:]
        #print("size exclusive: ",exclusive)

            # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return shared, exclusive

