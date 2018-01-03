import tensorflow as tf
import ops
import utils
from reader import Reader
from discriminator import Discriminator
from generatorEncoderDisen import GeneratorEncoderDisen
from generatorDecoder import GeneratorDecoder
#from generatorDecoderReversal import GeneratorDecoderReversal
import pdb

REAL_LABEL = 0.9

class CycleGANDisen:
  def __init__(self,
               X_train_file='',
               Y_train_file='',
               batch_size=1,
               image_size=256,
               use_lsgan=True,
               norm='instance',
               lambda1=1.0,
               lambda2=1.0,
               learning_rate=2e-4,
               beta1=0.5,
               ngf=64
              ):
    """
    Args:
      X_train_file: string, X tfrecords file for training
      Y_train_file: string Y tfrecords file for training
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
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.X_train_file = X_train_file
    self.Y_train_file = Y_train_file

    self.meanNoise = 0.0
    self.stddevNoise = 1.0

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.Ge = GeneratorEncoderDisen('Ge', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    self.Gd = GeneratorDecoder('Gd', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.Fe = GeneratorEncoderDisen('Fe', self.is_training, norm=norm, image_size=image_size)
    self.Fe = self.Ge
    self.Fd = GeneratorDecoder('Fd', self.is_training, norm=norm, image_size=image_size)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.Gdex = GeneratorDecoder('Gdex', self.is_training, norm=norm,
                                 image_size=image_size)

    self.Fdex = GeneratorDecoder('Fdex', self.is_training, norm=norm,
                                 image_size=image_size)

    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

  def model(self):
    X_reader = Reader(self.X_train_file, name='X',
        image_size=self.image_size, batch_size=self.batch_size)
    Y_reader = Reader(self.Y_train_file, name='Y',
        image_size=self.image_size, batch_size=self.batch_size)

    x = X_reader.feed()
    y = Y_reader.feed()


    # X -> Y
    rep_Sx, rep_Ex = self.Ge(x)
    noise = tf.random_normal(rep_Ex.shape, mean=self.meanNoise, stddev=self.stddevNoise)

    # Here, the exlusive bit comes before the shared part
    input_Gd = tf.concat([rep_Sx, noise],3)


    fake_y = self.Gd(input_Gd)
    G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)

    rep2_Sx, rep2_Ex = self.Fe(fake_y)
    noise = tf.random_normal(rep2_Ex.shape, mean=self.meanNoise, stddev=self.stddevNoise)

    input_Fd2 = tf.concat([noise, rep2_Sx],3)
    cycle_forward_loss = tf.reduce_mean(tf.abs(self.Fd(input_Fd2)-x))
    alignment_X_loss = tf.reduce_mean(tf.abs(self.Fd(tf.concat([rep_Ex,
                                                                rep_Sx],3))))
    # Reverse gradient layer as maximing reconstruction loss
    rev_X_loss = self.generator_loss(self.D_Y, self.Gdex(rep_Ex), use_lsgan=self.use_lsgan)


    #G_loss =  G_gan_loss + self.lambda1*cycle_forward_loss + self.lambda1*alignment_X_loss
    G_loss =  G_gan_loss + self.lambda1*cycle_forward_loss + self.lambda1*alignment_X_loss  + self.lambda1*rev_X_loss
    D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)

    # Y -> X
    rep_Sy, rep_Ey = self.Fe(y)
    noise = tf.random_normal(rep_Ey.shape, mean=self.meanNoise, stddev=self.stddevNoise)

    input_Fd = tf.concat([noise, rep_Sy],3)

    fake_x = self.Fd(input_Fd)
    F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)

    rep2_Sy, rep2_Ey = self.Ge(fake_x)
    noise = tf.random_normal(rep2_Ey.shape, mean=self.meanNoise, stddev=self.stddevNoise)

    input_Gd2 = tf.concat([rep2_Sy, noise],3)
    cycle_backward_loss = tf.reduce_mean(tf.abs(self.Gd(input_Gd2)-y))
    alignment_Y_loss = tf.reduce_mean(tf.abs(self.Gd(tf.concat([rep_Sy,
                                                                rep_Ey],3))))
    F_loss = F_gan_loss + self.lambda2*cycle_backward_loss + self.lambda2*alignment_Y_loss
    #F_loss = F_gan_loss + cycle_loss + alignment_loss
    D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)

    # summary
    tf.summary.histogram('D_Y/true', self.D_Y(y))
    tf.summary.histogram('D_Y/fake', self.D_Y(self.Gd(input_Gd)))
    tf.summary.histogram('D_X/true', self.D_X(x))
    tf.summary.histogram('D_X/fake', self.D_X(self.Fd(input_Fd)))

    tf.summary.histogram('RepX/exc', rep_Ex)
    tf.summary.histogram('RepX/gen', rep_Sx)
    tf.summary.histogram('RepX/noise', noise)

    tf.summary.scalar('loss/G_gan', G_gan_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/F_gan', F_gan_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/cycle_forward',  cycle_forward_loss)
    tf.summary.scalar('loss/cycle_backward',  cycle_backward_loss)
    tf.summary.scalar('loss/alignment_X', alignment_X_loss)
    tf.summary.scalar('loss/alignment_Y', alignment_Y_loss)

    tf.summary.image('X/generated',
                     utils.batch_convert2int(self.Gd(input_Gd)))
    tf.summary.image('X/cycle_rec',
                     utils.batch_convert2int(self.Fd(input_Fd2)))
    tf.summary.image('X/autoencoder_rec',
                     utils.batch_convert2int(self.Fd(tf.concat([rep_Ex, rep_Sx],3))))
    tf.summary.image('X/exclusive_rec',
                     utils.batch_convert2int(self.Gdex(rep_Ex)))


    tf.summary.image('Y/generated', utils.batch_convert2int(self.Fd(input_Fd)))
    tf.summary.image('Y/cycle_rec',
                     utils.batch_convert2int(self.Gd(input_Gd2)))
    tf.summary.image('Y/autoencoder_rec',
                     utils.batch_convert2int(self.Gd(tf.concat([rep_Sy,
                                                                rep_Ey],3))),max_outputs=3)
    #tf.summary.image('Y/exclusive_rec',
                     #utils.batch_convert2int(self.Fdex(rep_Ex)))

    #pdb.set_trace()
    tf.summary.image('Gen/X',
                     utils.batch_convert2fmint(rep_Sx,128),max_outputs=10)

    tf.summary.image('Ex/X',
                     utils.batch_convert2fmint(rep_Ex,64),max_outputs=10)

    tf.summary.image('Gen/Y',
                     utils.batch_convert2fmint(rep_Sy,128),max_outputs=10)

    tf.summary.image('Ex/Y',
                     utils.batch_convert2fmint(rep_Ey,64),max_outputs=10)




    return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
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

    G_optimizer = make_optimizer(G_loss, [self.Ge.variables, self.Gd.variables], name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, [self.Fe.variables, self.Fd.variables], name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

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

