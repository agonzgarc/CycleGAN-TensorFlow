import tensorflow as tf
import numpy as np
import ops
import utils
from reader_paired import ReaderPaired
from discriminator import Discriminator
from generatorEncoderDisen import GeneratorEncoderDisen
from generatorDecoder import GeneratorDecoder
from generatorDecoderReversal import GeneratorDecoderReversal
from domainClassifier import DomainClassifier
from generatorTransformer import GeneratorTransformer

import pdb

REAL_LABEL = 0.9

class PairedGANDisen:
  def __init__(self,
               XY_train_file='',
               batch_size=1,
               image_size=32,
               use_lsgan=True,
               norm='instance',
               lambdaRecon=1.0,
               lambdaAlign=1.0,
               lambdaRev=1.0,
               learning_rate=2e-4,
               beta1=0.5,
               nfS=64,
               nfE=36
              ):
    """
    Args:
      XY_train_file: string, X and Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      lambda1: integer, weight for forward cycle loss (X->Y->X)
      lambda2: integer, weight for backward cycle loss (Y->X->Y)
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      ngf: number of gen filters in first conv layer
    """
    self.lambdaRecon = lambdaRecon
    self.lambdaAlign = lambdaAlign
    self.lambdaRev = lambdaRev
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.XY_train_file = XY_train_file

    self.meanNoise = 0.0
    self.stddevNoise = 2.5

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.Ge = GeneratorEncoderDisen('Ge', self.is_training, norm=norm, image_size=image_size)
    self.Gd = GeneratorDecoder('Gd', self.is_training, norm=norm, image_size=image_size)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.Fe = GeneratorEncoderDisen('Fe', self.is_training,norm=norm, image_size=image_size)
    self.Fd = GeneratorDecoder('Fd', self.is_training, norm=norm, image_size=image_size)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    #self.Gdex = GeneratorDecoderReversal('Gdex', self.is_training, norm=norm,
                                 #image_size=image_size)

    #self.Fdex = GeneratorDecoderReversal('Fdex', self.is_training, norm=norm,
                                 #image_size=image_size)

    self.DC = DomainClassifier('DC', self.is_training, norm=norm)

    self.Gtex = GeneratorTransformer('Gtex', self.is_training, norm=norm, image_size=image_size)
    self.Ftex = GeneratorTransformer('Ftex', self.is_training, norm=norm, image_size=image_size)



    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

  def model(self):
    XY_reader = ReaderPaired(self.XY_train_file, name='XY',
        image_size=self.image_size, batch_size=self.batch_size)

    xy = XY_reader.feed()

    # Split returned batch into both domains
    x = xy[0]
    y = xy[1]

    # Generate representation with encoders
    # X -> Y
    rep_Sx, rep_Ex = self.Ge(x)
    # Y -> X
    rep_Sy, rep_Ey = self.Fe(y)

    # Compute stddevs of exclusive parts for noise generation
    mean_X, var_X = tf.nn.moments(rep_Ex, axes=[0,1,2])
    mean_Y, var_Y = tf.nn.moments(rep_Ey, axes=[0,1,2])

    noise = tf.random_normal(rep_Ey.shape, mean=mean_Y,
                                    stddev=tf.sqrt(var_Y))

    # Here, the exlusive bit comes before the shared part
    input_Gd = tf.concat([rep_Sx, noise],3)

    fake_y = self.Gd(input_Gd)
    G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)

    # Add reconstruction loss on shared features
    repR_Sx,_ = self.Fe(fake_y)
    X_features_loss = tf.reduce_mean(tf.abs(repR_Sx - rep_Sx))

    # For evaluation only, not used on optimization
    G_recon_loss = tf.reduce_mean(tf.abs(fake_y-y))

    # Reverse gradient layer as maximing gan loss from exclusive part
    #rev_X_loss = self.generator_loss(self.D_Y, self.Gdex(rep_Ex), use_lsgan=self.use_lsgan)

    # Transformation from exclusive to shared features
    GT_loss = tf.reduce_mean(tf.abs(self.Gtex(rep_Ex)-rep_Sx))


    G_loss =  G_gan_loss + GT_loss

    D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)

    noise = tf.random_normal(rep_Ex.shape, mean=mean_X,
                                    stddev=tf.sqrt(var_X))

    input_Fd = tf.concat([noise, rep_Sy],3)

    fake_x = self.Fd(input_Fd)
    F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)

    repR_Sy,_ = self.Ge(fake_x)
    Y_features_loss = tf.reduce_mean(tf.abs(repR_Sy - rep_Sy))

    F_recon_loss = tf.reduce_mean(tf.abs(fake_x-x))


    # Reverse gradient layer as maximing gan loss from exclusive part
    #rev_Y_loss = self.generator_loss(self.D_X, self.Fdex(rep_Ey), use_lsgan=self.use_lsgan)

    # Transformation from exclusive to shared features
    FT_loss = tf.reduce_mean(tf.abs(self.Ftex(rep_Ey)-rep_Sy))

    F_loss = F_gan_loss + FT_loss

    D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)


    # Alignment loss for autoencoders
    alignment_X_loss = tf.reduce_mean(tf.abs(self.Fd(tf.concat([rep_Ex,
                                                                rep_Sx],3))-x))
    alignment_Y_loss = tf.reduce_mean(tf.abs(self.Gd(tf.concat([rep_Sy,
                                                                rep_Ey],3))-y))

    # Add feature reconstruction loss to alignment as they work on same var set
    A_loss = alignment_X_loss + alignment_Y_loss
    Feat_loss = X_features_loss + Y_features_loss

    multiply = tf.constant([self.batch_size])
    dom_labels_x=tf.reshape(tf.tile(tf.constant([1.0,0.0]),multiply),[multiply[0],2])
    dom_labels_y=tf.reshape(tf.tile(tf.constant([0.0,1.0]),multiply),[multiply[0],2])
    dc_loss_x = self.domainClassifier_loss(self.DC,rep_Sx,dom_labels_x)
    dc_loss_y = self.domainClassifier_loss(self.DC,rep_Sy,dom_labels_y)

    DC_pred_X = tf.nn.softmax(self.DC(rep_Sx))
    DC_pred_Y = tf.nn.softmax(self.DC(rep_Sy))
    DC_loss = dc_loss_x + dc_loss_y

    # summary
    tf.summary.histogram('D_Y/true', self.D_Y(y))
    tf.summary.histogram('D_Y/fake', self.D_Y(self.Gd(input_Gd)))
    tf.summary.histogram('D_X/true', self.D_X(x))
    tf.summary.histogram('D_X/fake', self.D_X(self.Fd(input_Fd)))

    tf.summary.histogram('RepX/exc', rep_Ex)
    tf.summary.histogram('RepX/gen', rep_Sx)
    tf.summary.histogram('RepX/noise', noise)

    tf.summary.histogram('RepY/exc', rep_Ex)
    tf.summary.histogram('RepY/gen', rep_Sx)
    tf.summary.histogram('RepY/noise', noise)

    tf.summary.histogram('DC/X/scoreX', DC_pred_X[:,0])
    tf.summary.histogram('DC/X/scoreY', DC_pred_X[:,1])
    tf.summary.histogram('DC/Y/scoreX', DC_pred_Y[:,0])
    tf.summary.histogram('DC/Y/scoreY', DC_pred_Y[:,1])



    tf.summary.scalar('loss/G_total', G_loss)
    tf.summary.scalar('loss/G_gan', G_gan_loss)
    tf.summary.scalar('loss/GT_loss', GT_loss)
    tf.summary.scalar('loss/G_recon_loss',  G_recon_loss)
    #tf.summary.scalar('loss/G_rev_X_loss', rev_X_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/F_total', F_loss)
    tf.summary.scalar('loss/F_gan', F_gan_loss)
    tf.summary.scalar('loss/FT_total', FT_loss)
    tf.summary.scalar('loss/F_recon_loss',  F_recon_loss)
    #tf.summary.scalar('loss/F_rev_Y_loss', rev_Y_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/alignment_X', alignment_X_loss)
    tf.summary.scalar('loss/alignment_Y', alignment_Y_loss)
    tf.summary.scalar('loss/DC_loss_x', dc_loss_x)
    tf.summary.scalar('loss/DC_loss_y', dc_loss_y)
    tf.summary.scalar('loss/X_features_loss', X_features_loss)
    tf.summary.scalar('loss/Y_features_loss', Y_features_loss)



    tf.summary.image('X/generated',
                     utils.batch_convert2int(self.Gd(input_Gd)))

    noise2 = tf.random_normal(rep_Ey.shape, mean=mean_Y,
                                    stddev=tf.sqrt(var_Y))

    tf.summary.image('X/generated2',
                     utils.batch_convert2int(self.Gd(tf.concat([rep_Sx, noise2],3))))

    # swap representation
    #pdb.set_trace()
    ex1 = tf.reshape(rep_Ey[0,:],[1,rep_Ey.shape[1],rep_Ey.shape[2],rep_Ey.shape[3]])
    s1 = tf.reshape(rep_Sy[0,:],[1,rep_Sy.shape[1],rep_Sy.shape[2],rep_Sy.shape[3]])
    ex2 = tf.reshape(rep_Ey[1,:],[1,rep_Ey.shape[1],rep_Ey.shape[2],rep_Ey.shape[3]])
    s2 = tf.reshape(rep_Sy[1,:],[1,rep_Sy.shape[1],rep_Sy.shape[2],rep_Sy.shape[3]])

    ex3 = tf.reshape(rep_Ey[2,:],[1,rep_Ey.shape[1],rep_Ey.shape[2],rep_Ey.shape[3]])

    tf.summary.image('X/im1bk2',
                     utils.batch_convert2int(self.Gd(tf.concat([s1, ex2],3))))

    tf.summary.image('X/im2bk1',
                     utils.batch_convert2int(self.Gd(tf.concat([s2, ex1],3))))

    #tf.summary.image('X/sanitycheckim1bk1',
     #                utils.batch_convert2int(self.Gd(tf.concat([s1, ex1],3))))

    tf.summary.image('X/im2bk3',
                     utils.batch_convert2int(self.Gd(tf.concat([s2, ex3],3))))


    tf.summary.image('X/autoencoder_rec',
                     utils.batch_convert2int(self.Fd(tf.concat([rep_Ex, rep_Sx],3))))
    #tf.summary.image('X/exclusive_rec',
                     #utils.batch_convert2int(self.Gdex(rep_Ex)))


    tf.summary.image('Y/generated', utils.batch_convert2int(self.Fd(input_Fd)))
    tf.summary.image('Y/autoencoder_rec',
                     utils.batch_convert2int(self.Gd(tf.concat([rep_Sy,
                                                                rep_Ey],3))),max_outputs=3)
   # tf.summary.image('Y/exclusive_rec',
                     #utils.batch_convert2int(self.Fdex(rep_Ey)))



    return G_loss, D_Y_loss, F_loss, D_X_loss, A_loss, Feat_loss, DC_loss, fake_y, fake_x

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss, A_loss, Feat_loss, DC_loss):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer = make_optimizer(G_loss, [self.Ge.variables,self.Gd.variables,
                                         self.Gtex.variables], name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, [self.Fe.variables,
                                           self.Fd.variables, self.Ftex.variables], name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')
    A_optimizer = make_optimizer(A_loss,
                                 [self.Ge.variables,self.Gd.variables,self.Fe.variables,self.Fd.variables], name='Adam_A')
    DC_optimizer = make_optimizer(DC_loss, self.DC.variables, name='Adam_DC')
    Feat_optimizer = make_optimizer(Feat_loss,
                                 [self.Ge.variables,self.Gd.variables,self.Fe.variables,self.Fd.variables], name='Adam_Feat')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer,
                                  D_X_optimizer, A_optimizer, DC_optimizer]):
      return tf.no_op(name='optimizers')

  def domainClassifier_loss(self, DC, rep, dom):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=DC(rep),labels=dom))
      #loss = tf.reduce_mean(-tf.reduce_sum(dom*tf.log(DC(rep) + 1e-12), reduction_indices=[1]))
      return loss


  def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
      # use mean squared error
      error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(fake_y)))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss

  def cycle_consistency_loss(self, Ge, Gd, Fe, Fd, x, y):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(Fd(Fe(Gd(Ge(x))))-x))
    backward_loss = tf.reduce_mean(tf.abs(Gd(Ge((Fd(Fe(y)))))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss

  def alignment_loss(self, Ge, Gd, Fe, Fd, x, y):
    """ alignment loss (L1 norm)
    When in a separate function from the cycle_consistency_loss, slower as G_x
    and F_y need to be evaluated twice, merge!
    """
    # Combine the encoder of G and decoder of F
    x_alignment_loss = tf.reduce_mean(tf.abs(Fd(Ge(x))-x))

    # Combine the encoder of F and decoder of F
    y_alignment_loss = tf.reduce_mean(tf.abs(Gd(Fe(y))-y))
    loss = self.lambda1*x_alignment_loss + self.lambda2*y_alignment_loss
    return loss



  def sampleX2Y(self, input):
    """ Given input image, return generated output in the other
    domain
    """
    rep_Sx, rep_Ex = self.Ge(input)

    #mean_X, var_X = tf.nn.moments(rep_Ex, axes=[0,1,2])

    ##For now, make noise positive, as it takes the place of ReLU output
    #noise = tf.random_normal(rep_Ex.shape, mean=mean_X,
                                    #stddev=0.0*tf.sqrt(var_X))

    noise = tf.zeros(rep_Ex.shape)

    # Here, the exlusive bit comes before the shared part
    input_Gd = tf.concat([rep_Sx, noise],3)

    output_decoder = self.Gd(input_Gd);

    image = utils.batch_convert2int(output_decoder)
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))

    return image


  def sampleY2X(self, input):
    """ Given input image, return generated output in the other
    domain
    """
    rep_Sy, rep_Ey = self.Fe(input)

    # For now, make noise positive, as it takes the place of ReLU output
    #noise = tf.random_normal(rep_Ey.shape, mean=self.meanNoise,
                                    #stddev=self.stddevNoise)
    noise = tf.zeros(rep_Ey.shape)

    input_Fd = tf.concat([noise, rep_Sy],3)

    output_decoder= self.Fd(input_Fd)

    image = utils.batch_convert2int(output_decoder)
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))

    return image

  def swapExplicit(self, input1, input2):
    """ Given input image, return generated output in the other
    domain
    """
    rep1_Sy, rep1_Ey = self.Fe(input1)
    rep2_Sy, rep2_Ey = self.Fe(input2)


    input1_Fd = tf.concat([rep1_Sy, rep2_Ey],3)
    output1_decoder = self.Gd(input1_Fd)

    input2_Fd = tf.concat([rep2_Sy, rep1_Ey],3)
    output2_decoder = self.Gd(input2_Fd)

    image1 = utils.batch_convert2int(output1_decoder)
    image1 = tf.image.encode_jpeg(tf.squeeze(image1, [0]))

    image2 = utils.batch_convert2int(output2_decoder)
    image2 = tf.image.encode_jpeg(tf.squeeze(image2, [0]))

    return image1,image2




