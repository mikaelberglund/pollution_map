from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection
import numpy as np
#from data import generate_samples, preprocess

local_project_path = '/'
x_train = np.load("x_train.npy",)
x_train = np.expand_dims(x_train,axis=3)
y_train = np.load("y_train.npy")
x_train, x_test, y_train, y_test = model_selection.\
    train_test_split(x_train, y_train, test_size=.2)

model = models.Sequential()
model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(x_train.shape[1:4]), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

# history = model.fit_generator(
#     generate_samples(x_train, local_data_path),
#     samples_per_epoch=x_train.shape[0],
#     nb_epoch=30,
#     validation_data=generate_samples(x_test, local_data_path, augment=False),
# nb_val_samples = x_test.shape[0],
# )
history = model.fit(x_train, y_train, batch_size=2, epochs=3)
score = model.evaluate(x=x_test, y=y_test, batch_size=2, verbose=1)
print('Hi')