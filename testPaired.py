import tensorflow as tf
from modelPaired import PairedGANDisen
from reader_paired import ReaderPaired
from datetime import datetime
import os
import logging
from utils import ImagePool
import pdb

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 32, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 32, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambdaRecon', 0.0,
                        'weight for forward cycle loss (X->Y->X), default: 10.0')
tf.flags.DEFINE_integer('lambdaAlign', 1.0,
                        'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_integer('lambdaRev', 0.008,
                        'weight for backward cycle loss (Y->X->Y), default: 10.0')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('nfS', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('nfE', 36,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('XY', 'data/tfrecords/domain_MNIST_MNISTC.tfrecords',
                       'XY tfrecords file for training')
tf.flags.DEFINE_string('load_model', True,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
nameNet = 'mnistc_paired'

def test():
  checkpoints_dir = "checkpoints/{}".format(nameNet)
  if FLAGS.load_model is None:
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    paired_gan = PairedGANDisen(
        XY_train_file=FLAGS.XY,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambdaRecon=FLAGS.lambdaRecon,
        lambdaAlign=FLAGS.lambdaAlign,
        lambdaRev=FLAGS.lambdaRev,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        nfS=FLAGS.nfS,
        nfE=FLAGS.nfE
    )
    G_loss, D_Y_loss, F_loss, D_X_loss, A_loss, DC_loss, fake_y, fake_x = paired_gan.model()
    optimizers = paired_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss, A_loss, DC_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop():
        # get previously generated images
        fake_y_val, fake_x_val = sess.run([fake_y, fake_x])
        pdb.set_trace()

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  test()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
