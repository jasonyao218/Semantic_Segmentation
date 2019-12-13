from keras.models import *
from keras.layers import *
from keras.activations import *
import keras.backend as K
import keras

IMAGE_ORDERING = 'channels_last'


def encoder(input_height=416, input_width=416, pretrained='imagenet'):

	img_input = Input(shape=(input_height,input_width, 3))

	# Encoding by convolving and pooling for 5 times.
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
	f1 = x

	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	f2 = x

	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	f3 = x

	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	f4 = x 

	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
	f5 = x 

	return img_input , [f1 , f2 , f3 , f4 , f5 ]

def decoder(f, n_classes, n_up=3):
	o = f
	# Decoding by padding and upsampling.
	o = (ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)
	o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)
	for _ in range(n_up-2):
		o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
		o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
		o = (Conv2D(128, (3, 3), padding='valid' , data_format=IMAGE_ORDERING))(o)
		o = (BatchNormalization())(o)
	o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
	o = (ZeroPadding2D((1,1), data_format=IMAGE_ORDERING))(o)
	o = (Conv2D(64, (3, 3), padding='valid'  , data_format=IMAGE_ORDERING))(o)
	o = (BatchNormalization())(o)
	o =  Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)
	
	return o

def segnet(n_classes, input_height, input_width, encoder_level=3):
	img_input, levels = encoder(input_height, input_width)
	# Get features from encoded level.
	features = levels[encoder_level]
	# Decoding
	o = decoder(features, n_classes)
	# Reshape the result
	o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
	o = Softmax()(o)
	model = Model(img_input, o)
	model.model_name="segnet"
	return model




