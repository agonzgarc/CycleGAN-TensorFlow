import tensorflow as tf
import numpy as np
import ops
import utils
from reader_paired import ReaderPaired
from discriminator import Discriminator
from generatorEncoderDisen import GeneratorEncoderDisen
from generatorDecoder import GeneratorDecoder
from domainClassifier import DomainClassifier
from generatorExclusive import GeneratorExclusive
from discriminatorExclusive import DiscriminatorExclusive
import pdb

REAL_LABEL = 0.9

class PairedGANDisen:
  def __init__(self,
               XY_train_file='',
               batch_size=32,
               image_size=32,
               use_lsgan=True,
               norm='instance',
               learning_rate=2e-4,
               beta1=0.5,
               nfs=16,
               nfe=8
              ):
    """
    Args:
      XY_train_file: string, X and Y tfrecords file for training
      batch_size: integer, batch size
      image_size: integer, image size
      use_lsgan: boolean
      norm: 'instance' or 'batch'
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      nfS: size of the shared part of the representation
      nfE: size of the exclusive part of the representation
    """

    self.use_lsgan = use_lsgan
    use_sigmoid = not use_lsgan
    self.batch_size = batch_size
    self.image_size = image_size
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.XY_train_file = XY_train_file
    self.nfs = nfs
    self.nfe = nfe

    self.meanNoise = 0.0
    self.stddevNoise = 2.5

    self.weightGAN = 0.1

    self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

    self.Ge = GeneratorEncoderDisen('Ge', self.is_training, norm=norm,
                                    image_size=image_size, nfs=self.nfs, nfe=self.nfe)
    self.Gd = GeneratorDecoder('Gd', self.is_training, norm=norm, image_size=image_size, nfs=self.nfs, nfe=self.nfe)
    self.D_Y = Discriminator('D_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.Dex_Y = Discriminator('Dex_Y',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.Fe = GeneratorEncoderDisen('Fe', self.is_training,norm=norm, image_size=image_size, nfs=self.nfs, nfe=self.nfe)
    self.Fd = GeneratorDecoder('Fd', self.is_training, norm=norm, image_size=image_size, nfs=self.nfs, nfe=self.nfe)
    self.D_X = Discriminator('D_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.Dex_X = Discriminator('Dex_X',
        self.is_training, norm=norm, use_sigmoid=use_sigmoid)

    self.Gdex = GeneratorDecoder('Gdex', self.is_training, norm=norm,
                                 image_size=image_size, nfs=self.nfs,
                                 nfe=self.nfe, reverse=True)

    self.Fdex = GeneratorDecoder('Fdex', self.is_training, norm=norm,
                                 image_size=image_size, nfs=self.nfs,
                                 nfe=self.nfe, reverse=True)

    #self.Geex = GeneratorEncoderDisen('Geex',self.is_training, norm=norm,
                                      #image_size=image_size, nfs=0, nfe=self.nfe)
    #self.Feex = GeneratorEncoderDisen('Feex',self.is_training, norm=norm,
                                      #image_size=image_size, nfs=0, nfe=self.nfe)

    self.DC = DomainClassifier('DC', self.is_training, norm=norm)

    self.ZEx_Y = GeneratorExclusive('ZEx_Y', self.is_training, norm=norm, nfe=self.nfe)
    self.ZEx_X = GeneratorExclusive('ZEx_X', self.is_training, norm=norm, nfe=self.nfe)

    self.DZEx_Y = DiscriminatorExclusive('DZEx_Y', self.is_training, norm=norm, use_sigmoid=use_sigmoid)
    self.DZEx_X = DiscriminatorExclusive('DZEx_X', self.is_training, norm=norm, use_sigmoid=use_sigmoid)




    self.fake_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

    self.fake_ex_x = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])
    self.fake_ex_y = tf.placeholder(tf.float32,
        shape=[batch_size, image_size, image_size, 3])

    self.fake_Zex_x = tf.placeholder(tf.float32,
        shape=[batch_size, 8, 8, 16])
    self.fake_Zex_y = tf.placeholder(tf.float32,
        shape=[batch_size, 8, 8, 16])

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
    #mean_X, var_X = tf.nn.moments(rep_Ex, axes=[0,1,2])
    #mean_Y, var_Y = tf.nn.moments(rep_Ey, axes=[0,1,2])

    #### G block, X --> Y

    # Generate exlusive representation using noise
    noise_Y = tf.random_normal([self.batch_size, 1, 1, 10])
    fake_Zex_y = self.ZEx_Y(noise_Y)
    #ZEx_Y_loss = tf.reduce_mean(tf.abs(fake_Zex_y-rep_Ey))
    ZEx_Y_loss = self.generator_loss_1input(self.DZEx_Y, fake_Zex_y, use_lsgan=self.use_lsgan)

    DZEx_Y_loss = self.discriminator_loss_1input(self.DZEx_Y, rep_Ey,
                                                 self.fake_Zex_y, use_lsgan=self.use_lsgan)


    # Here, the exlusive bit comes before the shared part
    input_Gd = tf.concat([rep_Sx, fake_Zex_y],3)

    fake_y = self.Gd(input_Gd)
    G_gan_loss = self.generator_loss(self.D_Y, x, fake_y, use_lsgan=self.use_lsgan)


    # Add reconstruction loss on shared features
    #repR_Sx, repR_Ex = self.Fe(fake_y)
    #X_features_loss = tf.reduce_mean(tf.abs(repR_Sx - rep_Sx))

    #X_noise_loss = tf.reduce_mean(tf.abs(repR_Ex - noise))

    #X_features_noise_loss = X_features_loss + X_noise_loss

    #G_recon_loss = tf.reduce_mean(tf.abs(fake_y-y))

    # Reverse gradient layer as maximing gan loss from exclusive part
    fake_ex_y = self.Gdex(rep_Ex)

    #fake_ex_y = self.Gdex(rep_Sx,rep_Ex)
    Gdex_loss = self.generator_loss(self.Dex_Y, x, fake_ex_y, use_lsgan=self.use_lsgan)

    # Try to get the exclusive rep input with the image
    #_, repExc_Ex = self.Geex(fake_ex_y)
    #Geex_loss = tf.reduce_mean(tf.abs(repExc_Ex - rep_Ex))

    G_loss =  G_gan_loss + Gdex_loss# + Geex_loss
    G_loss = self.weightGAN*G_loss

    D_Y_loss = self.discriminator_loss(self.D_Y, x, y, self.fake_y, use_lsgan=self.use_lsgan)
    Dex_Y_loss = self.discriminator_loss(self.Dex_Y, x, y, self.fake_ex_y, use_lsgan=self.use_lsgan)



    #### F block, Y-->X
    # Generate exlusive representation using noise
    noise_X = tf.random_normal([self.batch_size, 1, 1, 10])
    fake_Zex_x = self.ZEx_X(noise_X)
    ZEx_X_loss = self.generator_loss_1input(self.DZEx_X, fake_Zex_x, use_lsgan=self.use_lsgan)

    DZEx_X_loss = self.discriminator_loss_1input(self.DZEx_X, rep_Ey,
                                                 self.fake_Zex_x, use_lsgan=self.use_lsgan)

    #ZEx_X_loss = tf.reduce_mean(tf.abs(fake_Zex_x-rep_Ex))

    ZEx_loss = ZEx_X_loss + ZEx_Y_loss

    input_Fd = tf.concat([fake_Zex_x, rep_Sy],3)

    fake_x = self.Fd(input_Fd)
    F_gan_loss = self.generator_loss(self.D_X, y, fake_x, use_lsgan=self.use_lsgan)

    #repR_Sy,_ = self.Ge(fake_x)
    #Y_features_loss = tf.reduce_mean(tf.abs(repR_Sy - rep_Sy))

    #F_recon_loss = tf.reduce_mean(tf.abs(fake_x-x))



    # Reverse gradient layer as maximing gan loss from exclusive part
    fake_ex_x = self.Fdex(rep_Ey)
    Fdex_loss = self.generator_loss(self.Dex_X, y, fake_ex_x, use_lsgan=self.use_lsgan)

    #_, repExc_Ey = self.Feex(fake_ex_y)
    #Feex_loss = tf.reduce_mean(tf.abs(repExc_Ey - rep_Ey))


    F_loss = F_gan_loss + Fdex_loss# + Feex_loss
    F_loss = self.weightGAN*F_loss
    D_X_loss = self.discriminator_loss(self.D_X, y, x, self.fake_x, use_lsgan=self.use_lsgan)
    Dex_X_loss = self.discriminator_loss(self.Dex_X, y, x, self.fake_ex_x, use_lsgan=self.use_lsgan)


    # Alignment loss for autoencoders
    alignment_X_loss = tf.reduce_mean(tf.abs(self.Fd(tf.concat([rep_Ex,
                                                                rep_Sx],3))-x))
    alignment_Y_loss = tf.reduce_mean(tf.abs(self.Gd(tf.concat([rep_Sy,
                                                                rep_Ey],3))-y))

    # Add feature reconstruction loss to alignment as they work on same var set
    A_loss = alignment_X_loss + 10*alignment_Y_loss

    #Feat_loss = X_features_loss + Y_features_loss

    # Feature reconstruction loss for the paired case
    Feat_loss = tf.reduce_mean(tf.abs(rep_Sx-rep_Sy))


    multiply = tf.constant([self.batch_size])
    dom_labels_x=tf.reshape(tf.tile(tf.constant([1.0,0.0]),multiply),[multiply[0],2])
    dom_labels_y=tf.reshape(tf.tile(tf.constant([0.0,1.0]),multiply),[multiply[0],2])
    dc_loss_x = self.domainClassifier_loss(self.DC,rep_Sx,dom_labels_x)
    dc_loss_y = self.domainClassifier_loss(self.DC,rep_Sy,dom_labels_y)

    DC_pred_X = tf.nn.softmax(self.DC(rep_Sx))
    DC_pred_Y = tf.nn.softmax(self.DC(rep_Sy))
    DC_loss = dc_loss_x + dc_loss_y

    # summary
    tf.summary.histogram('D_Y/true', self.D_Y(x,y))
    tf.summary.histogram('D_Y/fake', self.D_Y(x,self.Gd(input_Gd)))
    tf.summary.histogram('D_X/true', self.D_X(y,x))
    tf.summary.histogram('D_X/fake', self.D_X(y,self.Fd(input_Fd)))
    tf.summary.histogram('Dex_Y/true', self.Dex_Y(x,y))
    tf.summary.histogram('Dex_Y/fake', self.Dex_Y(x,self.Gdex(rep_Ex)))
    tf.summary.histogram('Dex_X/true', self.Dex_X(y,x))
    tf.summary.histogram('Dex_X/fake', self.Dex_X(y,self.Fdex(rep_Ey)))
    #tf.summary.histogram('DZex_Y/true', self.DZEx_Y(x,y))
    #tf.summary.histogram('Dex_Y/fake', self.Dex_Y(x,self.Gdex(rep_Ex)))
    #tf.summary.histogram('Dex_X/true', self.Dex_X(y,x))
    #tf.summary.histogram('Dex_X/fake', self.Dex_X(y,self.Fdex(rep_Ey)))



    tf.summary.histogram('RepX/exc', rep_Ex)
    tf.summary.histogram('RepX/gen', rep_Sx)
    tf.summary.histogram('RepX/noise', noise_X)

    tf.summary.histogram('RepY/exc', rep_Ey)
    tf.summary.histogram('RepY/gen', rep_Sy)
    tf.summary.histogram('RepY/noise', noise_Y)

    tf.summary.histogram('DC/X/scoreX', DC_pred_X[:,0])
    tf.summary.histogram('DC/X/scoreY', DC_pred_X[:,1])
    tf.summary.histogram('DC/Y/scoreX', DC_pred_Y[:,0])
    tf.summary.histogram('DC/Y/scoreY', DC_pred_Y[:,1])



    tf.summary.scalar('loss/G_total', G_loss)
    tf.summary.scalar('loss/G_gan', G_gan_loss)
    tf.summary.scalar('loss/Gdex_gan', Gdex_loss)
    tf.summary.scalar('loss/ZEx_loss', ZEx_loss)
    #tf.summary.scalar('loss/Geex_gan', Geex_loss)
    tf.summary.scalar('loss/D_Y', D_Y_loss)
    tf.summary.scalar('loss/Dex_Y', Dex_Y_loss)
    tf.summary.scalar('loss/DZEx_Y', DZEx_Y_loss)
    tf.summary.scalar('loss/F_total', F_loss)
    tf.summary.scalar('loss/F_gan', F_gan_loss)
    tf.summary.scalar('loss/Fdex_gan', Fdex_loss)
    #tf.summary.scalar('loss/Feex_gan', Feex_loss)
    tf.summary.scalar('loss/D_X', D_X_loss)
    tf.summary.scalar('loss/Dex_X', Dex_X_loss)
    tf.summary.scalar('loss/DZEx_X', DZEx_X_loss)
    tf.summary.scalar('loss/alignment_X', alignment_X_loss)
    tf.summary.scalar('loss/alignment_Y', alignment_Y_loss)
    tf.summary.scalar('loss/DC_loss_x', dc_loss_x)
    tf.summary.scalar('loss/DC_loss_y', dc_loss_y)
    #tf.summary.scalar('loss/X_features_loss', X_features_loss)
    #tf.summary.scalar('loss/Y_features_loss', Y_features_loss)
    tf.summary.scalar('loss/Feat_loss', Feat_loss)


    generatedX1 = self.Gd(input_Gd)
    tf.summary.image('X/generated',
                     utils.batch_convert2int(generatedX1 ))

    #noise2 = tf.random_normal(rep_Ey.shape, mean=mean_Y,
                                    #stddev=tf.sqrt(var_Y))

    noise2 = tf.random_normal([self.batch_size, 1, 1, 10])
    fake_Zex_y2 = self.ZEx_Y(noise2)

    generatedX2 = self.Gd(tf.concat([rep_Sx, fake_Zex_y2],3))
    tf.summary.image('X/generated2',
                     utils.batch_convert2int(generatedX2))

    noiseXVar = tf.reduce_mean(tf.abs(generatedX1[:,:4,:4,:] - generatedX2[:,:4,:4,:]))
    tf.summary.scalar('Eval/XnoiseVar',noiseXVar)



    # Autoencoders
    autoX = self.Fd(tf.concat([rep_Ex, rep_Sx],3))
    tf.summary.image('X/autoencoder_rec',
                     utils.batch_convert2int(autoX))
    tf.summary.image('X/exclusive_rec',
                     utils.batch_convert2int(self.Gdex(rep_Ex)))

    autoY = self.Gd(tf.concat([rep_Sy,rep_Ey],3))
    tf.summary.image('Y/autoencoder_rec',
                     utils.batch_convert2int(autoY),max_outputs=3)
    tf.summary.image('Y/exclusive_rec',
                     utils.batch_convert2int(self.Fdex(rep_Ey)))

    swapScoreBKG = self.computeSwapScoreBKG(rep_Sy, rep_Ey, autoY)

    # swap representation
    #ex1 = tf.reshape(rep_Ey[0,:],[1,rep_Ey.shape[1],rep_Ey.shape[2],rep_Ey.shape[3]])
    #s1 = tf.reshape(rep_Sy[0,:],[1,rep_Sy.shape[1],rep_Sy.shape[2],rep_Sy.shape[3]])
    #ex2 = tf.reshape(rep_Ey[1,:],[1,rep_Ey.shape[1],rep_Ey.shape[2],rep_Ey.shape[3]])
    #s2 = tf.reshape(rep_Sy[1,:],[1,rep_Sy.shape[1],rep_Sy.shape[2],rep_Sy.shape[3]])
    #ex3 = tf.reshape(rep_Ey[2,:],[1,rep_Ey.shape[1],rep_Ey.shape[2],rep_Ey.shape[3]])

    #im1bk2 = self.Gd(tf.concat([s1, ex2],3))
    #tf.summary.image('X/im1bk2',utils.batch_convert2int(im1bk2))

    #im2bk1 = self.Gd(tf.concat([s2, ex1],3))
    #tf.summary.image('X/im2bk1', utils.batch_convert2int(im2bk1))

    #im2bk3 = self.Gd(tf.concat([s2, ex3],3))
    #tf.summary.image('X/im2bk3', utils.batch_convert2int(im2bk3))

    ##Evaluation test on swapped background
    #swapScoreBKG = tf.reduce_mean(tf.abs(im1bk2[0,:4,:4,:] - autoY[1,:4,:4,:])) + tf.reduce_mean(tf.abs(im2bk1[0,:4,:4,:] - autoY[0,:4,:4,:]))
    tf.summary.scalar('Eval/swapScoreBKG', swapScoreBKG)


    tf.summary.image('Y/generated', utils.batch_convert2int(self.Fd(input_Fd)))

    # swap representation, X images
    ex1X = tf.reshape(rep_Ex[0,:],[1,rep_Ex.shape[1],rep_Ex.shape[2],rep_Ex.shape[3]])
    s1X = tf.reshape(rep_Sx[0,:],[1,rep_Sx.shape[1],rep_Sx.shape[2],rep_Sx.shape[3]])
    ex2X = tf.reshape(rep_Ex[1,:],[1,rep_Ex.shape[1],rep_Ex.shape[2],rep_Ex.shape[3]])
    s2X = tf.reshape(rep_Sx[1,:],[1,rep_Sx.shape[1],rep_Sx.shape[2],rep_Sx.shape[3]])

    im1bk2 =self.Fd(tf.concat([ex2X,s1X],3))
    tf.summary.image('Y/im1bk2', utils.batch_convert2int(im1bk2))

    im2bk1 = self.Fd(tf.concat([ex1X,s2X],3))
    tf.summary.image('Y/im2bk1', utils.batch_convert2int(im2bk1))

    im2bk0 = self.Fd(tf.concat([tf.zeros(ex1X.shape),s2X],3))
    tf.summary.image('Y/im2bkg0', utils.batch_convert2int(im2bk0))

    #pdb.set_trace()
    #autoX1 = tf.reshape(autoX[0,:,:,:],[1,32,32,3])
    #autoX2 = tf.reshape(autoX[1,:,:,:],[1,32,32,3])

    # Evaluation test on swapped background
    swapScoreFG = tf.reduce_mean(tf.abs(im1bk2[0,:,:,:] - autoX[0,:,:,:])) + tf.reduce_mean(tf.abs(im2bk1[0,:,:,:] - autoX[1,:,:,:]))

    tf.summary.scalar('Eval/swapScoreFG', swapScoreFG)


    # Show representation
    tf.summary.image('ZZExclRep/Xgenerated',
                     utils.batch_convert2fmint(rep_Ex,self.nfe),max_outputs=16)
    tf.summary.image('ZZExclRep/Xnoise', utils.batch_convert2fmint(fake_Zex_y,self.nfe),max_outputs=16)
    tf.summary.image('ZZExclRep/Ygenerated',
                     utils.batch_convert2fmint(rep_Ey,self.nfe),max_outputs=16)
    tf.summary.image('ZZExclRep/Ynoise', utils.batch_convert2fmint(fake_Zex_x,self.nfe),max_outputs=16)
    tf.summary.image('ZZSharedRep/X', utils.batch_convert2fmint(rep_Sx,self.nfs),max_outputs=4)
    tf.summary.image('ZZSharedRep/Y', utils.batch_convert2fmint(rep_Sy,self.nfs),max_outputs=4)


    # build dictionary to return
    loss_dict = {'G_loss':G_loss,
                 'D_Y_loss':D_Y_loss,
                 'Dex_Y_loss':Dex_Y_loss,
                 'DZEx_Y_loss':DZEx_Y_loss,
                 'F_loss':F_loss,
                 'D_X_loss':D_X_loss,
                 'Dex_X_loss':Dex_X_loss,
                 'DZEx_X_loss':DZEx_X_loss,
                 'A_loss':A_loss,
                 'Feat_loss':Feat_loss,
                 'DC_loss':DC_loss,
                 'ZEx_loss':ZEx_loss,
                 'fake_y':fake_y,
                 'fake_x':fake_x,
                 'fake_ex_y':fake_ex_y,
                 'fake_ex_x':fake_ex_x,
                 'fake_Zex_x':fake_Zex_x,
                 'fake_Zex_y':fake_Zex_y,
                 'swapScoreFG':swapScoreFG,
                 'swapScoreBKG':swapScoreBKG
                }
    return loss_dict

  def optimize(self, loss_dict):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 10000
      decay_steps = 10000
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
      #tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    optimizer_list = []
    if 'G_loss' in loss_dict :
        #G_optimizer = make_optimizer(loss_dict['G_loss'],
                                     #[self.Ge.variables,self.Gd.variables,self.Gdex.variables,self.Geex.variables], name='Adam_G')
        G_optimizer = make_optimizer(loss_dict['G_loss'], [self.Ge.variables,self.Gd.variables,self.Gdex.variables], name='Adam_G')
        optimizer_list.append(G_optimizer)

    if 'D_Y_loss' in loss_dict :
        D_Y_optimizer = make_optimizer(loss_dict['D_Y_loss'], self.D_Y.variables, name='Adam_D_Y')
        optimizer_list.append(D_Y_optimizer)

    if 'Dex_Y_loss' in loss_dict :
        Dex_Y_optimizer = make_optimizer(loss_dict['Dex_Y_loss'], self.Dex_Y.variables, name='Adam_Dex_Y')
        optimizer_list.append(Dex_Y_optimizer)

    if 'DZEx_Y_loss' in loss_dict :
        DZEx_Y_optimizer = make_optimizer(loss_dict['DZEx_Y_loss'],
                                          self.DZEx_Y.variables, name='Adam_DZEx_Y')
        optimizer_list.append(DZEx_Y_optimizer)

    if 'F_loss' in loss_dict :
        #F_optimizer =  make_optimizer(loss_dict['F_loss'], [self.Fe.variables, self.Fd.variables,self.Fdex.variables,self.Feex.variables], name='Adam_F')
        F_optimizer =  make_optimizer(loss_dict['F_loss'], [self.Fe.variables, self.Fd.variables,self.Fdex.variables], name='Adam_F')
        optimizer_list.append(F_optimizer)

    if 'D_X_loss' in loss_dict :
        D_X_optimizer = make_optimizer(loss_dict['D_X_loss'], self.D_X.variables, name='Adam_D_X')
        optimizer_list.append(D_X_optimizer)

    if 'Dex_X_loss' in loss_dict :
        Dex_X_optimizer = make_optimizer(loss_dict['Dex_X_loss'], self.Dex_X.variables, name='Adam_Dex_X')
        optimizer_list.append(Dex_X_optimizer)

    if 'DZEx_X_loss' in loss_dict :
        DZEx_X_optimizer = make_optimizer(loss_dict['DZEx_X_loss'],
                                          self.DZEx_X.variables, name='Adam_DZEx_X')
        optimizer_list.append(DZEx_X_optimizer)

    if 'A_loss' in loss_dict :
        A_optimizer = make_optimizer(loss_dict['A_loss'],
                                 [self.Ge.variables,self.Gd.variables,self.Fe.variables,self.Fd.variables], name='Adam_A')
        optimizer_list.append(A_optimizer)

    if 'Feat_loss' in loss_dict :
        Feat_optimizer = make_optimizer(loss_dict['Feat_loss'],
                                 [self.Ge.variables,self.Fe.variables], name='Adam_Feat')
        optimizer_list.append(Feat_optimizer)

    if 'ZEx_loss' in loss_dict :
        ZEx_optimizer = make_optimizer(loss_dict['ZEx_loss'],
                                 [self.ZEx_X.variables,self.ZEx_Y.variables], name='Adam_Feat')
        optimizer_list.append(Feat_optimizer)


    if 'DC_loss' in loss_dict :
        print("Setting DC optimizer")
        DC_optimizer = make_optimizer(loss_dict['DC_loss'], [self.DC.variables,
                                                            self.Ge.variables,
                                                            self.Fe.variables], name='Adam_DC')
        optimizer_list.append(DC_optimizer)


    with tf.control_dependencies(optimizer_list):
      return tf.no_op(name='optimizers')

  def domainClassifier_loss(self, DC, rep, dom):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=DC(rep),labels=dom))
      #loss = tf.reduce_mean(-tf.reduce_sum(dom*tf.log(DC(rep) + 1e-12), reduction_indices=[1]))
      return loss


  def discriminator_loss(self, D, x, y, fake_y, use_lsgan=True):
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
      error_real = tf.reduce_mean(tf.squared_difference(D(x,y), REAL_LABEL))
      error_fake = tf.reduce_mean(tf.square(D(x,fake_y)))
    else:
      # use cross entropy
      error_real = -tf.reduce_mean(ops.safe_log(D(y)))
      error_fake = -tf.reduce_mean(ops.safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss


  def discriminator_loss_1input(self, D, y, fake_y, use_lsgan=True):
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


  def generator_loss(self, D, x, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
      # use mean squared error
      loss = tf.reduce_mean(tf.squared_difference(D(x,fake_y), REAL_LABEL))
    else:
      # heuristic, non-saturating loss
      loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
    return loss

  def generator_loss_1input(self, D, fake_y, use_lsgan=True):
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

  def computeSwapScoreBKG(self, rep_Sy, rep_Ey, autoY):

    bkg_ims_idx = tf.random_uniform([self.batch_size],minval=0,maxval=self.batch_size,dtype=tf.int32)
    swapScoreBKG = 0
    #for i in range(0,3):
    for i in range(0,self.batch_size):
        s_curr = tf.reshape(rep_Sy[i,:],[1,rep_Sy.shape[1],rep_Sy.shape[2],rep_Sy.shape[3]])

        #print('I:'+str(i)+' paired with:'+str(bkg_ims_idx[i]))
        # Image to swap cannot be current image
        while bkg_ims_idx[i] == tf.Variable(i):
            pdb.set_trace()
            bkf_ims_idx[i] = tf.random_uniform([1],minval=0,maxval=self.batch_size,dtype=tf.int32)

        s_rnd = tf.reshape(rep_Sy[bkg_ims_idx[i],:],[1,rep_Sy.shape[1],rep_Sy.shape[2],rep_Sy.shape[3]])
        ex_rnd = tf.reshape(rep_Ey[bkg_ims_idx[i],:],[1,rep_Ey.shape[1],rep_Ey.shape[2],rep_Ey.shape[3]])
        im_swapped = self.Gd(tf.concat([s_curr,ex_rnd],3))

        # Only show first 3
        if i < 3:
            tf.summary.image('ZSwap/im_'+str(i)+'_iswapped',utils.batch_convert2int(im_swapped))
            tf.summary.image('ZSwap/im_'+str(i)+'_orig',utils.batch_convert2int(tf.reshape(autoY[bkg_ims_idx[i],:],
                                                                        [1,32,32,3])))
        swapScoreBKG += tf.reduce_mean(tf.abs(autoY[bkg_ims_idx[i],:4,:4,:] -
                                              im_swapped[0,:4,:4,:]))
    return swapScoreBKG

    #im1bk2 = self.Gd(tf.concat([s1, ex2],3))
    #tf.summary.image('X/im1bk2',utils.batch_convert2int(im1bk2))

    #im2bk1 = self.Gd(tf.concat([s2, ex1],3))
    #tf.summary.image('X/im2bk1', utils.batch_convert2int(im2bk1))

    #im2bk3 = self.Gd(tf.concat([s2, ex3],3))
    #tf.summary.image('X/im2bk3', utils.batch_convert2int(im2bk3))

    ## Evaluation test on swapped background
    #swapScoreBKG = tf.reduce_mean(tf.abs(im1bk2[0,:4,:4,:] - autoY[1,:4,:4,:])) + tf.reduce_mean(tf.abs(im2bk1[0,:4,:4,:] - autoY[0,:4,:4,:]))

