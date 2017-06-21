"""
    The ``image_loader`` module
    ======================
 
    Contains methods for loading images (patch and total)
"""


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os, os.path
import sys
import random


class Patch_loader :
    """
    Class Patch_loader
    ======================
 
    Defines loading functions for the patch database
    """

    def __init__(self, directory, size = 64, seed=42, 
    			 only_green=True) :
        # data init
        self.dir = directory          # directory with the train / test / validation sudirectories
        self.size = size              # size of the sub image that should be croped
        self.nb_channels = 3          # return only the green channel of the images
        if(only_green == True) :
            self.nb_channels = 1
        self.file_train = []          # list of the train images : tuple (image name / class)
        self.file_test = []           # list of the test images : tuple (image name / class)
        self.file_validation = []     # list of the validation images : tuple (image name / class)
        self.image_class = ['original', 'modified']         # list of the class (label) used in the process
        self.nb_class = len(self.image_class)
        self.train_iterator = 0       # iterator over the train images
        self.test_iterator = 0        # iterator over the test images
        self.validation_iterator = 0  # iterator over the validation images
        self.load_images(seed)        # load the data base


    def extract_channel(self, rgb_image, channel=1) :
        if channel > 2 :
            channel = 2
        return rgb_image[:,:,channel]

    def get_immediate_subdirectories(self,a_dir) :
        # return the list of sub directories of a directory
        return [name for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name))]

    def load_images_in_dir(self, dir_name, image_class) :

        # file extension accepted as image data
        proportion = self.proportion
        valid_image_extension = [".jpg", ".JPG", ".jpeg"]

        file_list = []

        for c in image_class :
            nb_image_per_class = 0
            file_list_by_class = []
            for filename in os.listdir(dir_name+'/'+c):
                # check if the file is an image
                extension = os.path.splitext(filename)[1]
                if extension.lower() in valid_image_extension:
                    file_list_by_class.append(filename)

            for i in range(int(len(file_list_by_class)*proportion)):
                file_list.append((file_list_by_class[i],c))
                nb_image_per_class += 1
            print('    ',c,nb_image_per_class,'images loaded')

        return file_list

    def load_images(self, seed) :

        # check if train / test / validation directories exists
        train_dir_name = self.dir + '/train'
        if not os.path.exists(train_dir_name):
            print("error: train directory does not exist")
            sys.exit(0)
            return

        validation_dir_name = self.dir + '/validation'
        if not os.path.exists(validation_dir_name):
            print("error: validation directory does not exist")
            sys.exit(0)
            return

        test_dir_name = self.dir + '/test'
        if not os.path.exists(test_dir_name):
            print("error: test directory does not exist")
            return []
            sys.exit(0)

        # count number of classes
        # self.image_class = self.get_immediate_subdirectories(train_dir_name)
        print('     number of classes :', self.nb_class, '   ', self.image_class)

        # load image file name and class
        print("\n     train data")
        self.file_train = self.load_images_in_dir(train_dir_name,self.image_class)
        print("\n     test data")
        self.file_test = self.load_images_in_dir(test_dir_name,self.image_class)
        print("\n     validation data")
        self.file_validation = self.load_images_in_dir(validation_dir_name,self.image_class)

        # shuffle the lists
        print("\n     shuffle lists ...")
        random.seed(seed)
        random.shuffle(self.file_train)
        random.shuffle(self.file_test)
        random.shuffle(self.file_validation)
        #print(self.file_train)

        #print("\n     loading done.")


	def get_next_image(self, directory, verbose = False) :

    	
		if directory not in set(['train', 'test', 'validation']):
			print("error: directory does not exist")
			return()
			sys.exit(0)

        # load next image (size should be big enough)
		image = []
        
        # pop file name and class
		if directory == 'train': 
			data = self.file_train[self.train_iterator]
			self.train_iterator += 1
			if self.train_iterator >= len(self.file_train) :
				self.train_iterator = 0
		if directory == 'test':
			data = self.file_test[self.test_iterator]
			self.test_iterator += 1
			if self.test_iterator >= len(self.file_test) :
				self.test_iterator = 0
		if directory == 'validation':
			data = self.file_validation[self.validation_iterator]
			self.validation_iterator += 1
			if self.validation_iterator >= len(self.file_validation) :
				self.validation_iterator = 0

       # load image
		file_name = self.dir + '/' + directory + '/' + data[1] + '/' + data[0]
		image = Image.open(file_name)
		if(verbose) :
			print("  ", file_name)
			print( '     index  :', self.train_iterator -1)
			print( '     width  :', image.size[0] )
			print( '     height :', image.size[1] )
			print( '     mode   :', image.mode    )
			print( '     format :', image.format  )

                
		image = np.asarray(image)

		if( self.nb_channels == 1 and len(image.shape) > 2 ) :
			image = self.extract_channel(image,1)
        # convert to float image
		image = image.astype(np.float32) / 255.
        #image = image.reshape(1, self.size, self.size, 3)

		image = image.reshape(self.size, self.size, self.nb_channels)

        # build class label
		label = np.zeros(len(self.image_class))
		pos = self.image_class.index(data[1])
		label[pos] = 1.0
        
        # return image and label
		return (image, label)

	def get_intrablock(self, image):

		if self.size%8 != 0: 
			print('Incorect size... Not divided by block size (8)') 
			return()

		blocks = []
		for k in range(self.nb_channels):
			for i in range(int(self.size/8)): 
				for j in range(int(self.size/8)):
					blocks.append(image[8*i:8*(i+1), 8*j:8*(j+1), k])

		blocks = np.array(blocks)
		return(blocks)

	def get_interblock(self, image):

		if self.size%8 != 0: 
			print('Incorect size... Not divided by block size (8)') 
			return()

		blocks = []
		for k in range(self.nb_channels):
			for i in range(int(self.size/8) - 1): 
				for j in range(int(self.size/8) - 1):
					blocks.append(image[4 + 8*i: 4 + 8*(i+1), 4 + 8*j: 4 + 8*(j+1), k])

		blocks = np.array(blocks)
		return(blocks)

	def get_next_train_batch(self, batch_size = 64): 
		nb_intra = int(self.size**2/(8**2)*self.nb_channels)
		nb_inter = int((self.size**2/(8**2) - 2*self.size/8 + 1)*self.nb_channels)
		batch_image_intra = np.empty([batch_size, self.size, self.size, nb_intra])
		batch_image_inter = np.empty([batch_size, self.size, self.size, nb_inter])
		batch_label = np.empty([batch_size, self.nb_class])
		for i in range(batch_size): 
			image, label = self.get_next_image(directory = 'train')
			batch_label[i] = label
			batch_image_intra[i] = get_intrablock(image)
			batch_image_inter[i] = get_interblock(image)
		return(batch_image_intra, batch_image_inter, batch_label)


	def get_next_test_batch(self, batch_size = 64): 
		nb_intra = int(self.size**2/(8**2)*self.nb_channels)
		nb_inter = int((self.size**2/(8**2) - 2*self.size/8 + 1)*self.nb_channels)
		batch_image_intra = np.empty([batch_size, self.size, self.size, nb_intra])
		batch_image_inter = np.empty([batch_size, self.size, self.size, nb_inter])
		batch_label = np.empty([batch_size, self.nb_class])
		for i in range(batch_size): 
			image, label = self.get_next_image(directory = 'test')
			batch_label[i] = label
			batch_image_intra[i] = get_intrablock(image)
			batch_image_inter[i] = get_interblock(image)
		return(batch_image_intra, batch_image_inter, batch_label)


	def get_next_validation_batch(self, batch_size = 64): 
		nb_intra = int(self.size**2/(8**2)*self.nb_channels)
		nb_inter = int((self.size**2/(8**2) - 2*self.size/8 + 1)*self.nb_channels)
		batch_image_intra = np.empty([batch_size, self.size, self.size, nb_intra])
		batch_image_inter = np.empty([batch_size, self.size, self.size, nb_inter])
		batch_label = np.empty([batch_size, self.nb_class])
		for i in range(batch_size): 
			image, label = self.get_next_image(directory = 'validation')
			batch_label[i] = label
			batch_image_intra[i] = get_intrablock(image)
			batch_image_inter[i] = get_interblock(image)
		return(batch_image_intra, batch_image_inter, batch_label)
