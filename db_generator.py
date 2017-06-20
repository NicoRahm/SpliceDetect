from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os, os.path
import sys
import random
import pickle

class Splicing_generator :
	"""
	Class Database_loader
	======================
 
	Defines a loading scheme for database
    """

	def __init__(self, input_directory, size_max = 1200, 
				 size_min = 200, seed=42, Q1 = 85, Q2 = 95):
		# data init
		self.input_directory = input_directory          # directory with the train / test / validation sudirectories
		self.size_max = size_max              # size of the sub image that should be croped
		self.size_min = size_min
		self.nb_channels = 3          # return only the green channel of the images
		self.files = []          	  # list of the train images : tuple (image name / class)
		self.iterator = 0       # iterator over the train images
		self.load_images(seed)        # load the data base
		self.seed = seed
		self.Q1 = Q1
		self.Q2 = Q2

	def load_images_in_dir(self, dir_name) :

	        # file extension accepted as image data
		valid_image_extension = [".png", ".tiff"]

		file_list = []
		nb_images = 0
		for filename in os.listdir(dir_name):
	        # check if the file is an image
			extension = os.path.splitext(filename)[1]
			if extension.lower() in valid_image_extension:
				file_list.append(filename)
				nb_images += 1
		print('    ',nb_images,'images loaded')

		return file_list

	def load_images(self, seed) :

	    # check if input directory exists
		if not os.path.exists(self.input_directory):
			print("error: input directory does not exist")
			sys.exit(0)
			return

		self.files= self.load_images_in_dir(self.input_directory)

	    # shuffle the lists
		print("\n     shuffle lists ...")
		random.seed(seed)
		random.shuffle(self.files)
		#print(self.file_train)

		print("\n     loading done.")


	def splice(self, image1, image2):

		shape1 = image1.shape
		shape2 = image2.shape
		adding = np.zeros(shape2)

		r = random.randint(self.size_min, self.size_max)
		while 2*r >= min(shape1[0], shape1[1], shape2[0], shape2[1]):
			r = random.randint(self.size_min, self.size_max)

		a1, b1 = random.randint(r, shape1[0] - r), random.randint(r, shape1[1] - r)
		                
		y,x = np.ogrid[-a1:shape1[0]-a1, -b1:shape1[1]-b1]
		mask1 = x*x + y*y <= r*r

		a2, b2 = random.randint(r, shape2[0] - r), random.randint(r, shape2[1] - r)                


		y,x = np.ogrid[-a2:shape2[0]-a2, -b2:shape2[1]-b2]
		mask2 = x*x + y*y <= r*r
		image2[mask2] = 0

		if adding[mask2].shape != image1[mask1].shape:
			return(None, None)

		adding[mask2] = image1[mask1]

		result = image2 + adding

		return(result, 128*mask2.astype(np.uint8))


	def export(self, target_directory, nb_images = 100): 

		# path to spliced images
		if not os.path.exists(target_directory + '/images'):
			os.mkdir(target_directory + '/images')
		images_path = target_directory + '/images/'

		# path to ground truth masks
		if not os.path.exists(target_directory + '/masks'):
			os.mkdir(target_directory + '/masks')
		masks_path = target_directory + '/masks/'

		# path to patch decomposition
		if not os.path.exists(target_directory + '/patch'):
			os.mkdir(target_directory + '/patch')
		patch_path = target_directory + '/patch/'
		if not os.path.exists(patch_path + '/original'):
			os.mkdir(patch_path + '/original')
		if not os.path.exists(patch_path + '/modified'):
			os.mkdir(patch_path + '/modified')	


		mod_pixel_prop = dict()
		for i in range(nb_images):

			print('Computing splicing ' + str(i+1) + '/' + str(nb_images))
			# getting the names of TIFF files 
			data1 = self.files[self.iterator]
			data2 = self.files[self.iterator + 1]
			self.iterator += 2
			if self.iterator >= len(self.files) - 1 :
				self.iterator = 0
				random.shuffle(self.files)

			# Opening TIFF files 
			file_name1 = self.input_directory + data1
			file_name2 = self.input_directory + data2
			tiff1 = Image.open(file_name1)
			tiff2 = Image.open(file_name2)

			# Compressing to JPEG with different qualities
			jpgname1 = '/tmp/image1.jpg'
			jpgname2 = '/tmp/image2.jpg'
			tiff1.save(jpgname1, "JPEG", quality=self.Q1)
			tiff2.save(jpgname2, "JPEG", quality=self.Q2)

			# Re-opening JPEG files
			image1 = np.array(Image.open(jpgname1))
			image2 = np.array(Image.open(jpgname2))

			# Computing splicing and mask
			splicing, mask = self.splice(image1, image2)
			# print(np.max(mask))

			if splicing is None: 
				print("Not generated...")
			else:
				# Save the splicing and the corresponding mask
				splicename = data1[:-5] + '_' + str(self.Q1) + '_' + data2[:-5] + '_' + str(self.Q2) + '.jpg'
				exp_splice = Image.fromarray(splicing.astype(np.uint8)).convert('RGB')
				exp_splice.save(images_path + splicename, "JPEG", quality=self.Q2)
				
				maskname = 'mask_' + data1[:-5] + '_' + str(self.Q1) + '_' + data2[:-5] + '_' + str(self.Q2) + '.jpg'
				exp_mask = Image.fromarray(mask)
				exp_mask.save(masks_path + maskname, "JPEG")

				# re-opening spliced image
				spliced_image = Image.open(images_path + splicename)
				patch_size = 64
				t = 0.3 # treshold for labeling 
				xsize, ysize = splicing.shape[1], splicing.shape[0]
				nx = int(xsize/patch_size)
				ny = int(ysize/patch_size)
				n_tot_patch = nx*ny 
				prop_mod_tot = np.mean(np.array(mask))/np.max(np.array(mask))

				n_patch_to_extract = int(n_tot_patch*prop_mod_tot)

				num_patch = 1
				nmod = 0
				nori = 0
				while num_patch < n_patch_to_extract:
					crop_width = random.randint(0,nx)*patch_size
					crop_height = random.randint(0, ny)*patch_size
					box = (crop_width, crop_height, crop_width+patch_size, crop_height+patch_size)
					patch = spliced_image.crop(box)
					patch_mask = np.array(exp_mask.crop(box))
					if np.max(patch_mask) == 0: 
						prop_mod = 0
					else:
						prop_mod = np.mean(patch_mask)/np.max(patch_mask)
					if prop_mod > t: 
						label = '/modified/'
						nmod += 1
					else: 
						label = '/original/'
						nori += 1
					# print(nori, nmod, n_patch_to_extract, num_patch)
					if (label == '/original/' and nori <= n_patch_to_extract/2) or (label == '/modified/' and nmod <= n_patch_to_extract/2):
						patch_name = splicename[:-4] + '_' + str(num_patch) + '.jpg'
						patch.save(patch_path + label + patch_name, "JPEG", quality=self.Q2)
						mod_pixel_prop[label+patch_name] = prop_mod
						num_patch += 1

		with open(target_directory + 'mod_pixel_prop' + '.pkl', 'wb') as f:
			pickle.dump(mod_pixel_prop, f, pickle.HIGHEST_PROTOCOL)				

if __name__ == '__main__':

	generator = Splicing_generator(input_directory = '/home/nicolas/Database_Splicing/Test/TIFF/',
								   Q1 = 60, Q2 = 95)

	generator.export(target_directory = '/home/nicolas/Database_Splicing/Test/splicing',
					 nb_images = 10)