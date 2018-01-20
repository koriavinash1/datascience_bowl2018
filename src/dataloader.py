import os
import sys
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Set some parameters
from config import (seed, batch_size, IMG_WIDTH, 
			IMG_HEIGHT, IMG_CHANNELS, 
			TRAIN_PATH, TEST_PATH)


# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1] # 670 folders
test_ids = next(os.walk(TEST_PATH))[1] # 65 folders

print "Number of train examples: {:.6f}".format(len(train_ids))
print "Number of test examples: {:.6f}".format(len(test_ids))

def pre_processing():
	pass

def load_data(ids):
	# Get and resize train images and masks
	X_train = []
	Y_train = []
	print('Getting and resizing train images and masks ... ')
	steps = []
	sys.stdout.flush()
	for id_ in tqdm(ids, total=len(ids)):
		path = TRAIN_PATH + id_
		img = cv2.imread(path + '/images/' + id_ + '.png')
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		X_train.append(img)
		shape = img.shape
		mask_ = np.zeros((shape[0], shape[1])) # channels = 1
		step = 0
		for mask_file in next(os.walk(path + '/masks/'))[2]:
			mask_ = mask_ + cv2.cvtColor(cv2.imread(path + '/masks/' + mask_file), cv2.COLOR_BGR2GRAY)
			step +=1
		Y_train.append(mask_)
		steps.append(step)
	return X_train, Y_train

class DataSet(object):
	def __init__(self):
		self._epochs_completed = 0
		self._index_in_epoch = 0

	def num_examples(self):
		return self._num_examples

	def epochs_completed(self):
		return self._epochs_completed

	def test_batch(self, batch_size):
		index = np.random.randint(0, len(test_ids), size=batch_size)
		ids = test_ids[index]
		X_test, Y_test = load_data(ids)
		return X_test, Y_test

	def train_batch(self, batch_size):
		self._index_in_epoch += batch_size

		if self._index_in_epoch >= len(train_ids): self._epochs_completed += 1

		index = np.random.randint(0, len(train_ids), size=batch_size, dtype="int32")
		ids = np.array(train_ids)[index]
		X_train, Y_train = load_data(ids)
		return {'x': X_train, 'y': Y_train}