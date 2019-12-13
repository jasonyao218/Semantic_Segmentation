from segnet import segnet
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import time
import keras
from keras import backend as K
import numpy as np

NCLASSES = 19
HEIGHT = 1024
WIDTH = 2048

def generate_arr(lines, batch_size):
	n = len(lines)
	i = 0
	while 1:
		train_x = []
		train_y = []
		for _ in range(batch_size):
			if i == 0:
				np.random.shuffle(lines)
			x = lines[i].split(' ')[0]
			y = lines[i].split(' ')[1].restrip()
			img_x = Image.open(x)
			img_x = img_x.resize((int(WIDTH), int(HEIGHT)))
			img_x = np.asarray(img_x)
			img_x = img_x/255
			train_x.append(x)
			img_y = Image.open(y)
			img_y = img_y.resize((int(WIDTH),int(HEIGHT)))
			img_y = np.asarray(img_y)
			img_y = np.expand_dims(img_y,axis=-1)
			labels = np.zeros((HEIGHT,WIDTH,NCLASSES))
			for c in range(NCLASSES):
				labels[:,:,c] = (img_y[:,:,0] == c).astype(int)
			labels = np.reshape(labels, (-1, NCLASSES))
			train_y.append(labels)
			i = (i+1) % n
		yield (np.array(train_x),np.array(train_y))

def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    loss = 4 * K.sum(crossloss)/HEIGHT/WIDTH
    return 

if __name__ == "main":
	log_dir = "../logs/"
	model = segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)

	weight_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', cache_subdir='models')
	model.load_weights(weight_path, by_name=True)

	with open('../img_seg.txt','r') as f:
		lines = f.readlines()

	# Shuffle lines
	np.random.seed(10101)
	np.random.shuffle(lines)
	np.random.seed(None)
	# Split into train and validation set
	nval = int(len(lines)*0.1)
	ntrain = len(lines) - nval

	checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
	reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
	model.compile(loss = loss, optimizer = Adam(lr=1e-3), metrics = ['accuracy'])
	batch_size = 4
	print('Train on {} samples, val on {} samples, with batch size {}.'.format(ntrain, nval, batch_size))
	model.fit_generator(generate_arr(lines[:ntrain], batch_size),
            steps_per_epoch=max(1, ntrain//batch_size),
            validation_data=generate_arrays_from_file(lines[ntrain:], batch_size),
            validation_steps=max(1, nval//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_learning_rate, early_stopping])
	model.save_weights(log_dir+'last1.h5')