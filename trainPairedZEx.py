import tensorflow as tf
from modelPairedZEx import PairedGANDisen
from reader_paired import ReaderPaired
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 32, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 32, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('nfs', 16,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('nfe', 16,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('XY', 'data/tfrecords/domain_MNIST_MNISTC.tfrecords',
                       'XY tfrecords file for training')
tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
nameNet = 'mnistc_paired_size'

def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    #current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(nameNet)
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
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        nfs=FLAGS.nfs,
        nfe=FLAGS.nfe
    )

    totalSteps = 40000

    #name_losses = [r'G_loss', 'D_Y_loss', 'Dex_Y_loss', 'F_loss', 'D_X_loss', 'Dex_X_loss', 'A_loss', 'Feat_loss', 'DC_loss']

    loss_dict = paired_gan.model()
    optimizers = paired_gan.optimize(loss_dict)


    #G_loss, D_Y_loss, Dex_Y_loss, F_loss, D_X_loss, Dex_X_loss, A_loss, Feat_loss, DC_loss, fake_y, fake_x, fake_ex_y, fake_ex_x = paired_gan.model()
    #optimizers = paired_gan.optimize(G_loss, D_Y_loss, Dex_Y_loss, F_loss,
                                     #D_X_loss, Dex_X_loss, A_loss, Feat_loss, DC_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(mta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      fake_ex_Y_pool = ImagePool(FLAGS.pool_size)
      fake_ex_X_pool = ImagePool(FLAGS.pool_size)

      fake_Zex_Y_pool = ImagePool(FLAGS.pool_size)
      fake_Zex_X_pool = ImagePool(FLAGS.pool_size)

      while not coord.should_stop():
        # get previously generated images
        fake_y_val, fake_x_val, fake_ex_y_val, fake_ex_x_val,fake_Zex_y_val, fake_Zex_x_val = sess.run([loss_dict['fake_y'], loss_dict['fake_x'],
                  loss_dict['fake_ex_y'], loss_dict['fake_ex_x'],
                  loss_dict['fake_Zex_y'], loss_dict['fake_Zex_x']])

        #train
        _, swapScoreFG_val, swapScoreBKG_val, summary = (
              sess.run(
                  [optimizers,
                   loss_dict['swapScoreFG'], loss_dict['swapScoreBKG'], summary_op],
                  feed_dict={paired_gan.fake_y: fake_Y_pool.query(fake_y_val),
                             paired_gan.fake_x: fake_X_pool.query(fake_x_val),
                             paired_gan.fake_ex_y:fake_ex_Y_pool.query(fake_ex_y_val),
                             paired_gan.fake_ex_x:fake_ex_X_pool.query(fake_ex_x_val),
                             paired_gan.fake_Zex_y:fake_Zex_Y_pool.query(fake_Zex_y_val),
                             paired_gan.fake_Zex_x:fake_Zex_X_pool.query(fake_Zex_x_val)}
              )
        )

        train_writer.add_summary(summary, step)
        train_writer.flush()

        if step % 100 == 0:
          logging.info('-----------Step %d:-------------' % step)
          logging.info('  swapScoreFG   : {}'.format(swapScoreFG_val))
          logging.info('  swapScoreBKG   : {}'.format(swapScoreBKG_val))

        if step % 10000 == 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
          logging.info("Model saved in file: %s" % save_path)

        if step == totalSteps:
            coord.request_stop()

        step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
