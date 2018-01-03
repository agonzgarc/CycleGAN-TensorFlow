import tensorflow as tf
import utils
import pdb

class ReaderPaired():
  def __init__(self, tfrecords_file, image_size=256,
    min_queue_examples=1000, batch_size=1, num_threads=8, name=''):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.tfrecords_file = tfrecords_file
    self.image_size = image_size
    self.min_queue_examples = min_queue_examples
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.reader = tf.TFRecordReader()
    self.name = name

  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      reader = tf.TFRecordReader()

      _, serialized_example = self.reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          features={
            'image1/file_name': tf.FixedLenFeature([], tf.string),
            'image1/encoded_image': tf.FixedLenFeature([], tf.string),
            'image2/file_name': tf.FixedLenFeature([], tf.string),
            'image2/encoded_image': tf.FixedLenFeature([], tf.string),
          })

      image1_buffer = features['image1/encoded_image']
      image1 = tf.image.decode_jpeg(image1_buffer, channels=3)
      image1 = self._preprocess(image1)
      image2_buffer = features['image2/encoded_image']
      image2 = tf.image.decode_jpeg(image2_buffer, channels=3)
      image2 = self._preprocess(image2)
      #pdb.set_trace()
      images = tf.train.shuffle_batch(
            [image1, image2], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size,
            min_after_dequeue=self.min_queue_examples
          )

    with tf.name_scope(self.name[0]):
      tf.summary.image('_input', images[0])

    with tf.name_scope(self.name[1]):
      tf.summary.image('_input', images[1])
      tf.summary.image('_average',tf.reshape(tf.reduce_mean(images[1],axis=0),[1,self.image_size,self.image_size,3]))
    return images

  def _preprocess(self, image):
    image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
    image = utils.convert2float(image)
    image.set_shape([self.image_size, self.image_size, 3])
    return image

def test_reader():
  TRAIN_FILE_1 = 'data/tfrecords/apple.tfrecords'
  TRAIN_FILE_2 = 'data/tfrecords/orange.tfrecords'

  with tf.Graph().as_default():
    reader1 = Reader(TRAIN_FILE_1, batch_size=2)
    reader2 = Reader(TRAIN_FILE_2, batch_size=2)
    images_op1 = reader1.feed()
    images_op2 = reader2.feed()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        batch_images1, batch_images2 = sess.run([images_op1, images_op2])
        print("image shape: {}".format(batch_images1))
        print("image shape: {}".format(batch_images2))
        print("="*10)
        step += 1
    except KeyboardInterrupt:
      print('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  test_reader()
