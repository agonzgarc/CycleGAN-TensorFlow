# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import pdb
# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_image_annotation_pairs_to_tfrecord_with_paths(filename_pairs, tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    index = 0
    for img_path, annotation_path,depth_path in filename_pairs:

        img_ = np.array(Image.open(img_path))
	img = img_[:,:240,:]
        annotation_ = np.array(Image.open(annotation_path))
	annotation = annotation_[:,:240]        
        depth_ = np.array(Image.open(depth_path))
	depth= depth_[:,:240]        
	depth = depth.astype('float32')
	depth = np.log(depth + 1.)
        index +=1
	print index

       
        # Unomment this one when working with surgical data
        # annotation = annotation[:, :, 0]

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = depth.shape[0]
        width = depth.shape[1]

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()
        depth_raw = depth.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
	    'image_name': _bytes_feature((img_path)),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw),
	    'depth_raw':_bytes_feature(depth_raw)}))

        writer.write(example.SerializeToString())

    writer.close()



def write_mnist_mnistc(mnist_train,mnistc_train, tfrecords_filename):

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    height = mnist_train[0].shape[0]
    width = mnist_train[0].shape[1]
    for k in xrange(len(mnist_train)/2):
        i = k
        j = k+len(mnist_train)/2

        mnist1,mnistc1= mnist_train[i].tostring(),mnistc_train[i].tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'mnist': _bytes_feature(mnist1),
            'mnistc': _bytes_feature(mnistm1)}))
        writer.write(example.SerializeToString())
        print('index: %d'%k)
    writer.close()




def read_mnist_mnistm_syn_shvn(tfrecord_filenames_queue):
    """Return image/annotation tensors that are created by reading tfrecord file.
    The function accepts tfrecord filenames queue as an input which is usually
    can be created using tf.train.string_input_producer() where filename
    is specified with desired number of epochs. This function takes queue
    produced by aforemention tf.train.string_input_producer() and defines
    tensors converted from raw binary representations into
    reshaped image/annotation tensors.
    Parameters
    ----------
    tfrecord_filenames_queue : tfrecord filename queue String queue object from tf.train.string_input_producer()
    
    Returns
    -------
    image, annotation : tuple of tf.int32 (image, annotation)
        Tuple of image/annotation tensors
    """
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(tfrecord_filenames_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'height1': tf.FixedLenFeature([], tf.int64),
        'width1': tf.FixedLenFeature([], tf.int64),
        'height2': tf.FixedLenFeature([], tf.int64),
        'width2': tf.FixedLenFeature([], tf.int64),
        'mnist1': tf.FixedLenFeature([], tf.string),
        'mnist2': tf.FixedLenFeature([], tf.string),
        'mnistm1': tf.FixedLenFeature([], tf.string),
        'mnistm2': tf.FixedLenFeature([], tf.string),
        'shvn1': tf.FixedLenFeature([], tf.string),
        'shvn2': tf.FixedLenFeature([], tf.string),
        'syn1': tf.FixedLenFeature([], tf.string),
        'syn2': tf.FixedLenFeature([], tf.string)
        })

    
    mnist1 = tf.decode_raw(features['mnist1'], tf.uint8)
    mnist2 = tf.decode_raw(features['mnist2'], tf.uint8)
    mnistm1 = tf.decode_raw(features['mnistm1'], tf.uint8)
    mnistm2 = tf.decode_raw(features['mnistm2'], tf.uint8)
    shvn1 = tf.decode_raw(features['shvn1'], tf.uint8)
    shvn2 = tf.decode_raw(features['shvn2'], tf.uint8)
    syn1 = tf.decode_raw(features['syn1'], tf.uint8)
    syn2 = tf.decode_raw(features['syn2'], tf.uint8)

    height1 = tf.cast(features['height1'], tf.int32)
    width1 = tf.cast(features['width1'], tf.int32)
    height2 = tf.cast(features['height2'], tf.int32)
    width2 = tf.cast(features['width2'], tf.int32)
    
    image1_shape = tf.stack([height1, width1, 3])
    image2_shape = tf.stack([height2, width2, 3])
    
    # The last dimension was added because
    # the tf.resize_image_with_crop_or_pad() accepts tensors
    # that have depth. We need resize and crop later.
    # TODO: See if it is necessary and probably remove third
    # dimension

    mnist1 = tf.reshape(mnist1, image1_shape)
    mnist2 = tf.reshape(mnist2, image1_shape)
    mnistm1 = tf.reshape(mnistm1, image1_shape)
    mnistm2 = tf.reshape(mnistm2, image1_shape)

    shvn1 = tf.reshape(shvn1,image2_shape)
    shvn2 = tf.reshape(shvn2,image2_shape)

    syn1 = tf.reshape(syn1,image2_shape)
    syn2 = tf.reshape(syn2,image2_shape)

    return mnist1,mnistm1,shvn1,syn1,mnist2,mnistm2,shvn2,syn2

