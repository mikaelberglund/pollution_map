from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection
from sklearn.neural_network import MLPRegressor
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from skimage.util import random_noise


def flipud_8D(x,y,x_temp,y_temp):
    x = np.append(x,x_temp[:, ::-1, :, :],0)
    y = np.append(y, y_temp, 0)
    return x,y
def fliplr_8D(x,y,x_temp,y_temp):
    x = np.append(x,x_temp[:, :, ::-1, :],0)
    y = np.append(y,y_temp,0)
    return x,y
def flipudlr_8D(x,y,x_temp,y_temp):
    x = np.append(x,x_temp[:, ::-1, ::-1, :],0)
    y = np.append(y_temp, y, 0)
    return x,y
def noise_8D(x,y,x_temp,y_temp):
    x = np.append(x,random_noise(x_temp),0)
    y = np.append(y_temp, y, 0)
    return x,y

if False:
    filelist = glob.glob('Pollution_8D/*.pkl')
    dfm_files = list(filter(lambda k: 'dfm' in k, filelist))
    dfs_files = list(filter(lambda k: 'dfs' in k, filelist))
    dfm = pd.read_pickle(dfm_files[0])
    dfs = pd.read_pickle(dfs_files[0])
    for f in dfm_files[1:]:
        dfm = dfm.append(pd.read_pickle(f))
    for f in dfs_files[1:]:
        dfs = dfs.append(pd.read_pickle(f))
    print('Hi')
if True:
    filelist = glob.glob('Pollution_8D/*.npy')
    x_files = list(filter(lambda k: 'x_train' in k, filelist))
    y_files = list(filter(lambda k: 'y_train' in k, filelist))
    x_files.sort()
    y_files.sort()
    x_train = np.load(x_files[0])
    y_train = np.load(y_files[0])
    t = 1
    for f in x_files[t:]:
        if np.shape(np.load(f))[1:] == (23, 45, 8):
            x_train = np.append(x_train, np.load(f), axis=0)
            y_train = np.append(y_train, np.load(y_files[t]), axis=0)
        else:
            print('Error: Train file has wrong dimensions: '+str(f))
        t += 1
    # for f in y_files[1:]:
    #     y_train = np.append(y_train, np.load(f), axis=0)
local_project_path = '/'

print('Number of training images: '+str(x_train.shape[0]))
### FILLNA IN ALL IMAGES
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)
### OUTLIER REMOVAL
df = pd.DataFrame([x_train.mean(axis=1).mean(axis=1)[:,0],y_train]).T
df = df.rename({0:'x',1:'y'},axis='columns')
quantiles = 0.1
xin = np.logical_and(df.x >= np.quantile(a=df.x,q=quantiles),df.x <= np.quantile(a=df.x,q=1-quantiles))
xin = np.logical_and(xin,df.x != 0)
yin = np.logical_and(df.y >= np.quantile(a=df.y,q=quantiles),df.y <= np.quantile(a=df.y,q=1-quantiles))
yin = np.logical_and(yin,df.y != 0)
f = np.array(xin&yin)
x_train = x_train[f]
y_train = y_train[f]

### NORMALIZE
xmax = x_train.max()
ymax = y_train.max()
x_train = x_train/xmax
y_train = y_train/ymax
print('Number of training images (excl. outliers): '+str(x_train.shape[0]))
x_train, x_test, y_train, y_test = model_selection.\
    train_test_split(x_train, y_train, test_size=.1)

### AUGMENT DATA
x_temp = x_train
y_temp = y_train
x_train, y_train = flipud_8D(x_train,y_train,x_temp,y_temp)
x_train, y_train = fliplr_8D(x_train,y_train,x_temp,y_temp)
x_train, y_train = flipudlr_8D(x_train,y_train,x_temp,y_temp)
x_train, y_train = noise_8D(x_train,y_train,x_temp,y_temp)
x_temp1, y_temp1 = flipud_8D(x_train,y_train,x_temp,y_temp)
x_train, y_train = noise_8D(x_temp1,y_temp1,x_temp,y_temp)
x_temp1, y_temp1 = fliplr_8D(x_train,y_train,x_temp,y_temp)
x_train, y_train = noise_8D(x_temp1,y_temp1,x_temp,y_temp)
x_temp1, y_temp1 = flipudlr_8D(x_train,y_train,x_temp,y_temp)
x_train, y_train = noise_8D(x_temp1,y_temp1,x_temp,y_temp)
print('Number of training images (after augmentation, excl. outliers): '+str(x_train.shape[0]))

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ' x len is '+ str(x_train.shape[0])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = models.Sequential()
model.add(convolutional.Convolution2D(4, 3, 3, input_shape=(x_train.shape[1:4]), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
# model.add(core.Dense(100, activation='sigmoid'))
# model.add(core.Dropout(0.2))
# model.add(core.Dense(30, activation='relu'))
# model.add(core.Dropout(0.2))
model.add(core.Dense(1))
#model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
model.compile(optimizer=optimizers.SGD(), loss='mean_squared_error')
# history = model.fit(it, steps_per_epoch=313)

history = model.fit(x_train, y_train,validation_split=0.1, batch_size=30, epochs=30, callbacks=[tensorboard_callback])
#score = model.evaluate(x=x_test, y=y_test, batch_size=30, verbose=1)
pred = model.predict(x_test)
#ax = sns.regplot(x=y_test*ymax,y=pred[:,0]*ymax)
ax = sns.regplot(x=y_test,y=pred[:,0])
ax.set(xlabel="Truth", ylabel = "Prediction")
print('Hi')


# history = model.fit_generator(
#     generate_samples(x_train, local_data_path),
#     samples_per_epoch=x_train.shape[0],
#     nb_epoch=30,
#     validation_data=generate_samples(x_test, local_data_path, augment=False),
# nb_val_samples = x_test.shape[0],
# )
