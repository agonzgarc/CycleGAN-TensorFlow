import tensorflow as tf
import random
import os
import pdb

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data0_input_dir', '/home/abel/Datasets/MNIST/images/',
                       'X input directory, default: data/apple2orange/trainB')

tf.flags.DEFINE_string('data1_input_dir', '/home/abel/Datasets/MNISTC/images/',
                       'X input directory, default: data/apple2orange/trainB')


tf.flags.DEFINE_string('data0_output_file',
                       'data/tfrecords/domain_MNIST_MNISTC.tfrecords',
                       'X output tfrecords file, default: data/tfrecords/orange.tfrecords')


def data_reader(input_dir, shuffle=False):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  """Build an Example proto for an example.
  Args:
    file_path: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  file_name = file_path.split('/')[-1]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def data_writer(input1_dir,input2_dir, output_file):
  """Write data to tfrecords
  """
  file1_paths = data_reader(input1_dir)
  file2_paths = data_reader(input2_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error, e:
    pass

  images_num = len(file1_paths)

  # Both datasets need to have the same number of images
  assert images_num == len(file2_paths)

  # dump to tfrecords file
  writer = tf.python_io.TFRecordWriter(output_file)

  for i in range(images_num):
    file1_path = file1_paths[i]
    file2_path = file2_paths[i]

    with tf.gfile.FastGFile(file1_path, 'rb') as f:
      image1_data = f.read()

    with tf.gfile.FastGFile(file2_path, 'rb') as f:
      image2_data = f.read()


    file1_name = file1_path.split('/')[-1]
    file2_name = file2_path.split('/')[-1]

    example = tf.train.Example(features=tf.train.Features(feature={
      'image1/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file1_name))),
      'image1/encoded_image': _bytes_feature((image1_data)),
      'image2/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file2_name))),
      'image2/encoded_image': _bytes_feature((image2_data))
    }))

    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))

  print("Done.")
  writer.close()

def main(unused_argv):
  print("Saving data0 and data1 to tfrecords...")
  data_writer(FLAGS.data0_input_dir, FLAGS.data1_input_dir, FLAGS.data0_output_file)

if __name__ == '__main__':
  tf.app.run()
