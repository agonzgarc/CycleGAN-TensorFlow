"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from modelPaired import PairedGANDisen
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'pretrained/swapAutStd.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input1', 'inputs/input_sample.png', 'input image path (.jpg)')
tf.flags.DEFINE_string('input2', 'inputs/input_sample.png', 'input image path (.jpg)')
tf.flags.DEFINE_string('output1', 'outputs/output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_string('output2', 'outputs/output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '32', 'image size, default: 256')

def inference():
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(FLAGS.input1, 'rb') as f:
      image_data = f.read()
      input_image1 = tf.image.decode_jpeg(image_data, channels=3)
      input_image1 = tf.image.resize_images(input_image1, size=(FLAGS.image_size, FLAGS.image_size))
      input_image1 = utils.convert2float(input_image1)
      input_image1.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

    with tf.gfile.FastGFile(FLAGS.input2, 'rb') as f:
      image_data = f.read()
      input_image2 = tf.image.decode_jpeg(image_data, channels=3)
      input_image2 = tf.image.resize_images(input_image2, size=(FLAGS.image_size, FLAGS.image_size))
      input_image2 = utils.convert2float(input_image2)
      input_image2.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())

    [output_image1,output_image2] = tf.import_graph_def(graph_def,
                          input_map={'input_image1':
                                     input_image1,'input_image2':input_image2},
                          return_elements=['output_image1:0','output_image2:0'],
                          name='output')

  with tf.Session(graph=graph) as sess:
    generated1 = output_image1.eval()
    with open(FLAGS.output1, 'wb') as f:
      f.write(generated1)

    generated2 = output_image2.eval()
    with open(FLAGS.output2, 'wb') as f:
      f.write(generated2)

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()
