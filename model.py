"""
    The ``model`` module
    ======================
 
    Contains the class Model which implements the core model for splicing detection, 
    training, testing and visualization functions.
"""

import os

import time
import random
from . import image_loader as il
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import configparser

import numpy as np

import csv

GPU = '/gpu:0'
config = 'server'

# seed initialisation
print("\n   random initialisation ...")
random_seed = int(time.time() % 10000 ) 
random.seed(random_seed)  # for reproducibility
print('   random seed =', random_seed)

# tool functions

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape, nb_input, seed = None):
  """Creates and initializes (truncated normal distribution) a variable weight Tensor with a defined shape"""
  sigma = np.sqrt(2/nb_input)
  initial = tf.truncated_normal(shape, stddev=sigma, seed = random_seed)
  return tf.Variable(initial)

def bias_variable(shape):
  """Creates and initializes (truncated normal distribution with 0.5 mean) a variable bias Tensor with a defined shape"""
  initial = tf.truncated_normal(shape, mean = 0.5, stddev=0.1, seed = random_seed)
  return tf.Variable(initial)
  
def conv2d(x, W):
  """Returns the 2D convolution between input x and the kernel W"""  
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool_2x2(x):
  """Returns the result of max-pooling on input x with a 2x2 window""" 
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
  """Returns the result of average-pooling on input x with a 2x2 window""" 
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

def max_pool_10x10(x):
  """Returns the result of max-pooling on input x with a 10x10 window""" 
  return tf.nn.max_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')

def avg_pool_10x10(x):
  """Returns the result of average-pooling on input x with a 10x10 window""" 
  return tf.nn.avg_pool(x, ksize=[1, 10, 10, 1],
                           strides=[1, 10, 10, 1], padding='SAME')


class Model:

  """
    Class Model
    ======================
 
    Defines a model for single-image CG detection and numerous methods to : 
    - Create the TensorFlow graph of the model
    - Train the model on a specific database
    - Reload past weights 
    - Test the model (simple classification, full-size images with boosting and splicing)
    - Visualize some images and probability maps
"""

  def __init__(self, database_path, config = 'Personal', filters_intra = [32, 64], 
               filters_inter = [32, 64], batch_size = 64, only_green = True, 
               using_GPU = False):
    """Defines a model for splicing detection

    """ 
    clear = lambda: os.system('clear')
    clear()
    print('   tensorFlow version: ', tf.__version__)
    
    # read the configuration file
    conf = configparser.ConfigParser()
    conf.read('config.ini')

    if config not in conf:
      raise ValueError(config + ' is not in the config.ini file... Please create the corresponding section')
    
    self.dir_ckpt = conf[config]['dir_ckpt']
    self.dir_summaries = conf[config]['dir_summaries']
    self.dir_visualization = conf[config]['dir_visualization']
    print('   Check-points directory : ' + self.dir_ckpt)
    print('   Summaries directory : ' + self.dir_summaries)
    print('   Visualizations directory : ' + self.dir_visualization)

    # setting the parameters of the model
    self.nf_intra = filters_intra
    self.nf_inter = filters_inter
    self.nl_intra = len(self.nf_intra)
    self.nl_inter = len(self.nf_inter)
    self.filter_size = 3

    self.database_path = database_path
    self.image_size = 64
    self.batch_size = batch_size
    self.using_GPU = using_GPU
    self.nb_class = 2
    self.only_green = only_green

    # getting the database
    self.import_database()

    self.nb_channels = self.data.nb_channels

    # create the TensorFlow graph
    if using_GPU:
      with tf.device(GPU):
        self.create_graph(nb_class = self.nb_class,nf_intra = self.nf_intra, 
                          nf_inter = self.nf_inter, filter_size = self.filter_size)
    else: 
      self.create_graph(nb_class = self.nb_class,nf_intra = self.nf_intra, 
                        nf_inter = self.nf_inter, filter_size = self.filter_size)



  def import_database(self): 
    """Creates a Database_loader to load images from the distant database"""

    # load data
    print('   import data : image_size = ' + 
        str(self.image_size) + 'x' + str(self.image_size) + '...')
    self.data = il.Database_loader(self.database_path, self.image_size, 
                                   proportion = 1, only_green=self.only_green)

  def create_graph(self, nb_class, nf_intra = [32, 64], nf_inter = [32, 64], filter_size = 3): 
    """Creates the TensorFlow graph"""
    nl_intra = len(nf_intra)
    nl_inter = len(nf_inter)
    print('   create model ...')
    # input layer. One entry is a float size x size, 3-channels image. 
    # None means that the number of such vector can be of any lenght.

    graph = tf.Graph()

    with graph.as_default():

      with tf.name_scope('Input_Data'):
        nb_intra = int(self.image_size**2/(8**2)*self.nb_channels)
        nb_inter = int((self.image_size**2/(8**2) - 2*self.image_size/8 + 1)*self.nb_channels)
        x_intra = tf.placeholder(tf.float32, [None, 8, 8, nb_intra], name = 'x_intra')
        x_inter = tf.placeholder(tf.float32, [None, 8, 8, nb_inter], name = 'x_inter')
        self.x_intra = x_intra
        self.x_inter = x_inter        

      # Intra-block Net
      print('   Creating Intra-block Net')
      # first conv layer
      print('   Creating layer 1 - Shape : ' + str(self.filter_size) + 'x' + 
            str(self.filter_size) + 'x' + str(nb_intra) + 'x' + str(nf_intra[0]))

      with tf.name_scope('Intrablock_Conv1'):

        with tf.name_scope('Weights'):
          W_conv1 = weight_variable([self.filter_size, self.filter_size, nb_intra, nf_intra[0]],
                                     nb_input = 8*8*nb_intra, 
                                     seed = random_seed)
        with tf.name_scope('Bias'):
          b_conv1 = bias_variable([nf_intra[0]])


        # relu on the conv layer
        h_conv1 = tf.nn.relu(conv2d(x_intra, W_conv1) + b_conv1, 
                             name = 'Activated_1')

      self.W_convs_intra = [W_conv1]
      self.b_convs_intra = [b_conv1]
      self.h_convs_intra = [h_conv1]
      for i in range(1, nl_intra):
        print('   Creating layer ' + str(i+1) + ' - Shape : ' + str(self.filter_size) + 'x' + 
            str(self.filter_size) + 'x' + str(nf_intra[i-1]) + 'x' + str(nf_intra[i]))
        # other conv 
        with tf.name_scope('Intrablock_Conv' + str(i+1)):
          with tf.name_scope('Weights'):
            W_conv2 = weight_variable([self.filter_size, self.filter_size, nf_intra[i-1], nf_intra[i]],
                                      nb_input = 8*8*nf_intra[i-1])
            self.W_convs_intra.append(W_conv2)
          with tf.name_scope('Bias'):
            b_conv2 = bias_variable([nf_intra[i]])
            self.b_convs_intra.append(b_conv2)

          h_conv2 = tf.nn.relu(conv2d(self.h_convs_intra[i-1], W_conv2) + b_conv2, 
                               name = 'Activated_2')

          self.h_convs_intra.append(h_conv2)    

      print('   Creating MaxPool 2x2')
      with tf.name_scope('Intrablock_MaxPool'):
        self.maxpool_intra = max_pool_2x2(self.h_convs_intra[-1])


      # Inter-block Net
      print('   Creating Inter-block Net')
      # first conv layer
      print('   Creating layer 1 - Shape : ' + str(self.filter_size) + 'x' + 
            str(self.filter_size) + 'x' + str(nb_inter) + 'x' + str(nf_inter[0]))

      with tf.name_scope('Interblock_Conv1'):

        with tf.name_scope('Weights'):
          W_conv1 = weight_variable([self.filter_size, self.filter_size, nb_inter, nf_inter[0]], 
                                    nb_input = 8*8*nb_inter, seed = random_seed)
        with tf.name_scope('Bias'):
          b_conv1 = bias_variable([nf_inter[0]])


        # relu on the conv layer
        h_conv1 = tf.nn.relu(conv2d(x_inter, W_conv1) + b_conv1, 
                             name = 'Activated_1')

      self.W_convs_inter = [W_conv1]
      self.b_convs_inter = [b_conv1]
      self.h_convs_inter = [h_conv1]
      for i in range(1, nl_inter):
        print('   Creating layer ' + str(i+1) + ' - Shape : ' + str(self.filter_size) + 'x' + 
            str(self.filter_size) + 'x' + str(nf_inter[i-1]) + 'x' + str(nf_inter[i]))
        # other conv 
        with tf.name_scope('Interblock_Conv' + str(i+1)):
          with tf.name_scope('Weights'):
            W_conv2 = weight_variable([self.filter_size, self.filter_size, nf_inter[i-1], nf_inter[i]],
                                      nb_input = 8*8*nf_inter[i-1])
            self.W_convs_inter.append(W_conv2)
          with tf.name_scope('Bias'):
            b_conv2 = bias_variable([nf_inter[i]])
            self.b_convs_inter.append(b_conv2)

          h_conv2 = tf.nn.relu(conv2d(self.h_convs_inter[i-1], W_conv2) + b_conv2, 
                               name = 'Activated_2')

          self.h_convs_intra.append(h_conv2)  

      print('   Creating MaxPool 2x2')
      with tf.name_scope('Interblock_MaxPool'):
        self.maxpool_inter = max_pool_2x2(self.h_convs_inter[-1])


      print('   Merging networks')
      
      size_flat_intra = nf_intra[-1]*16
      flatten_intra = tf.reshape(self.maxpool_intra, [-1, size_flat_intra], name = "Flattened_intra")
      size_flat_inter = nf_inter[-1]*16
      flatten_inter = tf.reshape(self.maxpool_inter, [-1, size_flat_inter], name = "Flattened_inter")

      self.flatten = tf.concat([flatten_intra, flatten_inter], 1, name = 'Flatten_features')

      size_flat = size_flat_inter + size_flat_intra


      print('   Creating MLP ')
      # Densely Connected Layer
      # we add a fully-connected layer with 1024 neurons 
      with tf.variable_scope('Dense1'):
        with tf.name_scope('Weights'):
          W_fc1 = weight_variable([size_flat, 1024],
                                  nb_input = size_flat)
        with tf.name_scope('Bias'):
          b_fc1 = bias_variable([1024])
        # put a relu
        h_fc1 = tf.nn.relu(tf.matmul(self.flatten, W_fc1) + b_fc1, 
                           name = 'activated')

      # dropout
      with tf.name_scope('Dropout1'):
        keep_prob = tf.placeholder(tf.float32)
        self.keep_prob = keep_prob
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      self.h_fc1 = h_fc1

      # readout layer
      with tf.variable_scope('Readout'):
        with tf.name_scope('Weights'):
          W_fc3 = weight_variable([1024, nb_class], 
                                  nb_input = 1024)
        with tf.name_scope('Bias'):
          b_fc3 = bias_variable([nb_class])
        y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

      self.y_conv = y_conv

      # support for the learning label
      y_ = tf.placeholder(tf.float32, [None, nb_class])
      self.y_ = y_



      # Define loss (cost) function and optimizer
      print('   setup loss function and optimizer ...')

      # softmax to have normalized class probabilities + cross-entropy
      with tf.name_scope('cross_entropy'):

        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv)
        with tf.name_scope('total'):
          cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy)

      tf.summary.scalar('cross_entropy', cross_entropy_mean)

      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)

      self.train_step = train_step
      print('   test ...')
      # 'correct_prediction' is a function. argmax(y, 1), here 1 is for the axis number 1
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

      # 'accuracy' is a function: cast the boolean prediction to float and average them
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('training_accuracy', accuracy)

      self.accuracy = accuracy

    self.graph = graph
    print('   model created.')


  def train(self, nb_train_batch, nb_test_batch, 
            nb_validation_batch, validation_frequency = 50, show_filters = False):
    """
    Trains the model on the selected database training set.
    """
    run_name = input("   Choose a name for the run : ")
    path_save = self.dir_ckpt + run_name
    acc_name = self.dir_summaries + run_name + "/validation_accuracy_" + run_name + ".csv"


    # computation time tick
    start_clock = time.clock()
    start_time = time.time()
    batch_clock = None

    # start a session
    print('   start session ...')
    with tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=self.using_GPU)) as sess:

      merged = tf.summary.merge_all()
      
      if not os.path.exists(self.dir_summaries + run_name):
        os.mkdir(self.dir_summaries + run_name)


      train_writer = tf.summary.FileWriter(self.dir_summaries + run_name,
                                           sess.graph)

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      saver = tf.train.Saver()
      print('   variable initialization ...')

      restore_weigths = input("\nRestore weight from previous session ? (y/N) : ")

      if restore_weigths == 'y':
        file_to_restore = input("\nName of the file to restore (Directory : " + 
                                self.folder_ckpt + ') : ')
        saver.restore(sess, self.folder_ckpt + file_to_restore)
        print('\n   Model restored\n')
        

      # Train
      print('   train ...')
      start_clock = time.clock()
      start_time = time.time()
      validation_accuracy = []
      for i in range(nb_train_batch):
        
          # evry validation_frequency batches, test the accuracy
          if i%validation_frequency == 0 :
              
              if i%100 == 0:
                plot_histograms = False
              else:
                plot_histograms = False

              v = self.validation_testing(i, train_writer, nb_iterations = nb_validation_batch, 
                                      batch_size = self.batch_size, 
                                      plot_histograms = plot_histograms,
                                      run_name = run_name,
                                      show_filters = show_filters)
              validation_accuracy.append(v)
              
          # regular training
          batch = self.data.get_next_train_batch(self.batch_size)
          feed_dict = {self.x_intra: batch[0], self.x_inter: batch[1], self.y_: batch[2], self.keep_prob: 0.65}
          summary, _ = sess.run([merged, self.train_step], feed_dict = feed_dict)
          train_writer.add_summary(summary, i)

          # Saving weights every 100 batches
          if i%100 == 0:

            path_save_batch = path_save + str(i) + ".ckpt"
            print('   saving weights in file : ' + path_save_batch)
            saver.save(sess, path_save_batch)
            print('   OK')
            if batch_clock is not None: 
              time_elapsed = (time.time()-batch_clock)
              print('   Time last 100 batchs : ', time.strftime("%H:%M:%S",time.gmtime(time_elapsed)))
              remaining_time = time_elapsed * int((nb_train_batch - i)/100)
              print('   Remaining time : ', time.strftime("%H:%M:%S",time.gmtime(remaining_time)))
            batch_clock = time.time()
      
      print('   saving validation accuracy...')
      file = open(acc_name, 'w', newline='')

      try:
          writer = csv.writer(file)
       
          for v in validation_accuracy:
            writer.writerow([str(v)])
      finally:

          file.close()
          print('   done.')

      if nb_train_batch > validation_frequency:
        plt.figure()
        plt.plot(np.linspace(0,nb_train_batch,int(nb_train_batch/10)), validation_accuracy)
        plt.title("Validation accuracy during training")
        plt.xlabel("Training batch")
        plt.ylabel("Validation accuracy")
        plt.show()
        plt.close()
    # final test
      print('   final test ...')
      test_accuracy = 0
      # test_auc = 0
      nb_iterations = 20
      self.data.test_iterator = 0
      for _ in range( nb_iterations ) :
          batch_test = self.data.get_batch_test(self.batch_size, False, True, True)
          feed_dict = {self.x:batch_test[0], self.y_: batch_test[1], self.keep_prob: 1.0}
          test_accuracy += self.accuracy.eval(feed_dict)
          # test_auc += sess.run(auc, feed_dict)[0]

                
      test_accuracy /= nb_iterations
      print("   test accuracy %g"%test_accuracy)

      # test_auc /= (nb_iterations - 1)
      # print("   test AUC %g"%test_auc)

    # done
    print("   computation time (cpu) :",time.strftime("%H:%M:%S", time.gmtime(time.clock()-start_clock)))
    print("   computation time (real):",time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
    print('   done.')


  def validation_testing(self, it, writer, nb_iterations = 20, batch_size = 64,
                         plot_histograms = False, range_hist = [0.,1.], 
                         selected_hist_nb = 8, run_name = '',
                         show_filters = True):
    """
    Computes validation accuracy during training and plots some visualization.
    """
    if show_filters: 
      
      nb_height = 4
      nb_width = int(self.nb_conv1/nb_height)

      img, axes = plt.subplots(nrows = nb_width, ncols = nb_height)
      gs1 = gridspec.GridSpec(nb_height, nb_width)
      for i in range(self.nb_conv1):
        ax1 = plt.subplot(gs1[i])
        ax1.axis('off')
        im = plt.imshow(self.W_conv1[:,:,0,i].eval(), cmap = 'jet')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([]) 
        # axes.get_yaxis().set_ticks([])
        # plt.ylabel('Kernel ' + str(i), fontsize = 5.0)
        # ax1.set_ylabel('Kernel ' + str(i), fontsize = 5.0)
        ax1.set_title("Filter " + str(i + 1), fontsize = 12.0)    

      img.subplots_adjust(wspace = 0.1, hspace = 0.6, right = 0.7)
      cbar_ax = img.add_axes([0.75, 0.15, 0.03, 0.7])
      cbar = img.colorbar(im, ticks=[-0.5, 0, 0.5], cax=cbar_ax)
      cbar.ax.set_yticklabels(['< -0.5', '0', '> 0.5'])
      plt.show(img)
      plt.close()     


    validation_batch_size = batch_size 
    validation_accuracy = 0
    # validation_auc = 0
    self.data.validation_iterator = 0

    for _ in range( nb_iterations ) :
      batch_validation = self.data.get_batch_validation(batch_size=validation_batch_size, 
                                                        crop = False, 
                                                        random_flip_flop = True, 
                                                        random_rotate = True)
      feed_dict = {self.x_intra: batch_validation[0], 
                   self.x_inter: batch_validation[1],
                   self.y_: batch_validation[2], 
                   self.keep_prob: 1.0}
      validation_accuracy += self.accuracy.eval(feed_dict)

    validation_accuracy /= nb_iterations

    summary = tf.Summary()
    summary.value.add(tag="ValidationAccuracy", simple_value=validation_accuracy)

    writer.add_summary(summary, it)
    print("     step %d, training accuracy %g (%d validations tests)"%(it, validation_accuracy, validation_batch_size*nb_iterations))
    return(validation_accuracy)


if __name__ == '__main__': 


  database_path = '/home/nicolas/Database/'

  model = Model(database_path = database_path, config = 'Server', filters_intra = [64*3, 128], 
                filters_inter = [49*3, 128], batch_size = 64, only_green = False, using_GPU = False)

  model.train(nb_train_batch = 2000, 
              nb_test_batch = 100,
              nb_validation_batch = 20)