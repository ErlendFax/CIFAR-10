# Plot ad hoc CIFAR10 instances
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print X_train.shape, y_train.shape

def get_model():
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(3,32,32),output_shape=(3,32,32)))
	model.add(Conv2D(16, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
 	model.add(Conv2D(32, 3, 3, subsample=(2, 2), border_mode="same"))
 	#model.add(ELU())
 	#model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
 	model.add(Flatten())
 	model.add(Dropout(.2))
 	model.add(ELU())
 	model.add(Dense(512))
 	model.add(Dropout(.5))
 	model.add(ELU())
 	model.add(Dense(10, activation='softmax'))
 	model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
 	return model

model = get_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#for i in range(0, 9):
#	pyplot.subplot(330 + 1 + i)
#	pyplot.imshow(toimage(X_train[i+10*i/2]))
# show the plot
#pyplot.show()
