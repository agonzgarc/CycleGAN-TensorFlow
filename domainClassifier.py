import tensorflow as tf
import ops
import utils

@tf.RegisterGradient("ReverseGrad")
def _reverse_grad(unused_op, grad):
    return -1.0*grad

class DomainClassifier:
  def __init__(self, name, is_training, norm='instance', image_size=32):
    self.name = name
    self.reuse = False
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x num_feat_maps
    Returns:
      output: one unit with domain tag (0/1)
    """
    with tf.variable_scope(self.name):
        #g = tf.get_default_graph()
        #with g.gradient_override_map({"Identity": "ReverseGrad"}):
            idinput = tf.stop_gradient(input)
            #idinput = tf.identity(input)
            fc_output = ops.fully_connected(idinput, reuse=self.reuse, name='fc',units=100)

            output = ops.logits(fc_output, reuse=self.reuse, name ='logits')

    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output
