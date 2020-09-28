from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection
from sklearn.neural_network import MLPRegressor
import numpy as np
import glob
import pandas as pd
import seaborn as sns

if False:
    filelist = glob.glob('Pollution/*.pkl')
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
    filelist = glob.glob('Pollution/*.npy')
    x_files = list(filter(lambda k: 'x_train' in k, filelist))
    y_files = list(filter(lambda k: 'y_train' in k, filelist))
    x_train = np.load(x_files[0])

    y_train = np.load(y_files[0])
    for f in x_files[1:]:
        x_train = np.append(x_train, np.load(f), axis=0)
    for f in y_files[1:]:
        y_train = np.append(y_train, np.load(f), axis=0)
    x_train = np.expand_dims(x_train, axis=3)
local_project_path = '/'
# x_train = np.load("x_train.npy",)
# x_train = np.expand_dims(x_train,axis=3)
# y_train = np.load("y_train.npy")
print('Number of training images: '+str(x_train.shape[0]))
### FILLNA IN ALL IMAGES
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)
### OUTLIER REMOVAL
df = pd.DataFrame([x_train.mean(axis=1).mean(axis=1)[:,0],y_train]).T
df = df.rename({0:'y',1:'x'},axis='columns')
xin = df.x.between(df.x.quantile(.025), df.x.quantile(.975))
yin = df.y.between(df.y.quantile(.025), df.y.quantile(.975))
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

# x = x_train[:,:,:,0].reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# xtest = x_test[:,:,:,0].reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
# regr = MLPRegressor(random_state=1, max_iter=500).fit(x, y_train)
# regr.predict(xtest[:2])

model = models.Sequential()
model.add(convolutional.Convolution2D(4, 3, 3, input_shape=(x_train.shape[1:4]), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
# model.add(core.Dense(40, activation='relu'))
# model.add(core.Dropout(0.5))
#model.add(core.Dense(15, activation='relu'))
model.add(core.Dense(15, activation='sigmoid'))
model.add(core.Dropout(0.3))
model.add(core.Dense(1))
#model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
model.compile(optimizer=optimizers.SGD(), loss='mean_squared_error')

history = model.fit(x_train, y_train,validation_split=0.1, batch_size=30, epochs=30)
#score = model.evaluate(x=x_test, y=y_test, batch_size=30, verbose=1)
pred = model.predict(x_test)
sns.regplot(x=y_test,y=pred[:,0])
print('Hi')


# history = model.fit_generator(
#     generate_samples(x_train, local_data_path),
#     samples_per_epoch=x_train.shape[0],
#     nb_epoch=30,
#     validation_data=generate_samples(x_test, local_data_path, augment=False),
# nb_val_samples = x_test.shape[0],
# )
