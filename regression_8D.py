from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling
from sklearn import model_selection
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from skimage.util import random_noise
import matplotlib.pyplot as plt
from sklearn import linear_model, tree
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from keras.layers import ReLU, ReLU


from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start() #Start tracking energy use.
def flipud_8D(x,y,x_temp,y_temp,f,di):
    x_temp = x_temp[f].reshape(np.insert(x_temp.shape[1:4], 0, di, axis=0))
    x = np.append(x,x_temp[:, ::-1, :, :],0)
    y = np.append(y, y_temp[f[:,0,0,0]], 0)
    return x,y
def fliplr_8D(x,y,x_temp,y_temp,f,di):
    x_temp = x_temp[f].reshape(np.insert(x_temp.shape[1:4], 0, di, axis=0))
    x = np.append(x,x_temp[:, :, ::-1, :],0)
    y = np.append(y,y_temp[f[:,0,0,0]],0)
    return x,y
def flipudlr_8D(x,y,x_temp,y_temp,f,di):
    x_temp = x_temp[f].reshape(np.insert(x_temp.shape[1:4], 0, di, axis=0))
    x = np.append(x,x_temp[:, ::-1, ::-1, :],0)
    y = np.append(y_temp[f[:,0,0,0]], y, 0)
    return x,y
def noise_8D(x,y,x_temp,y_temp,f,di):
    x_temp = x_temp[f].reshape(np.insert(x_temp.shape[1:4], 0, di, axis=0))
    x = np.append(x,random_noise(x_temp),0)
    y = np.append(y_temp[f[:,0,0,0]], y, 0)
    return x,y

path = 'Pollution_8DXL/*.npy'
# path = 'Pollution_8D/*.npy'
if False:
    filelist = glob.glob(path)
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
    filelist = glob.glob(path)
    x_files = list(filter(lambda k: 'x_train' in k, filelist))
    y_files = list(filter(lambda k: 'y_train' in k, filelist))
    x_files.sort()
    y_files.sort()
    t = 0
    #while np.load(x_files[t]).shape[1:] != (23, 45, 8):
    while np.load(x_files[t]).shape[1:] != (32, 64, 8):
        t += 1
    x_train = np.load(x_files[t])
    y_train = np.load(y_files[t])
    for f in x_files[t+1:]:
        #if np.shape(np.load(f))[1:] == (23, 45, 8):
        if np.shape(np.load(f))[1:] == (32, 64, 8):
            x_train = np.append(x_train, np.load(f), axis=0)
            y_train = np.append(y_train, np.load(y_files[t+1]), axis=0)
        else:
            print('Error: Train file has wrong dimensions: '+str(f))
        t += 1
local_project_path = '/'

print('Number of training images: '+str(x_train.shape[0]))
### FILLNA IN ALL IMAGES
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)
### OUTLIER REMOVAL
if True:
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

### REBALANCE DATA
if True:
    for i in range(0,1):
        d = pd.DataFrame(y_train)
        iv = pd.cut(d[0],200).value_counts().sort_index().index[0]
        f = ~(d[0].between(iv.left,iv.right).values)
        keep = pd.cut(d[0], 200).value_counts()[1]
        f[0:2 * keep] = True
        di = f.sum()
        #f = np.reshape(np.broadcast_to(f,np.append(x_train.shape[1:4],x_train.shape[0])),x_train.shape)
        f = np.broadcast_to(f,x_train.shape[::-1]).T
        x_train = x_train[f].reshape(np.insert(x_train.shape[1:4], 0, di, axis=0))
        y_train = y_train[f[:,0,0,0]]
    

### STANDARDIZE DATA
standardize = True
if standardize:
    x_train = (x_train - np.mean(x_train))/np.std(x_train)
    y_train = (y_train - np.mean(y_train))/np.std(y_train)

### NORMALIZE
xmax = x_train.max()
ymax = y_train.max()
x_train = x_train/xmax
y_train = y_train/ymax
print('Number of training images (excl. outliers): '+str(x_train.shape[0]))

### SPLIT INTO TRAIN & TEST
x_train, x_test, y_train, y_test = model_selection.\
    train_test_split(x_train, y_train, test_size=.2)

### AUGMENT DATA
x_temp = x_train
y_temp = y_train
if False:
    d = pd.DataFrame(y_train)
    iv = pd.cut(d[0],2).value_counts().sort_index().index[0]
    f = ~(d[0].between(iv.left,iv.right).values)
    di = f.sum()
    f = np.reshape(np.broadcast_to(f,np.append(x_train.shape[1:4],x_train.shape[0])),x_train.shape)
    x_train = x_train[f].reshape(np.insert(x_train.shape[1:4], 0, di, axis=0))
    y_train = y_train[f[:,0,0,0]]
else:
    f = np.full(shape=x_train.shape,fill_value=True)
    di = x_train.shape[0]
x_train, y_train = flipud_8D(x_train,y_train,x_temp,y_temp,f,di)
x_train, y_train = fliplr_8D(x_train,y_train,x_temp,y_temp,f,di)
x_train, y_train = flipudlr_8D(x_train,y_train,x_temp,y_temp,f,di)
x_train, y_train = noise_8D(x_train,y_train,x_temp,y_temp,f,di)
x_temp1, y_temp1 = flipud_8D(x_train,y_train,x_temp,y_temp,f,di)
x_train, y_train = noise_8D(x_temp1,y_temp1,x_temp,y_temp,f,di)
x_temp1, y_temp1 = fliplr_8D(x_train,y_train,x_temp,y_temp,f,di)
x_train, y_train = noise_8D(x_temp1,y_temp1,x_temp,y_temp,f,di)
x_temp1, y_temp1 = flipudlr_8D(x_train,y_train,x_temp,y_temp,f,di)
x_train, y_train = noise_8D(x_temp1,y_temp1,x_temp,y_temp,f,di)
if False:
    d = pd.DataFrame(y_train)
    iv = pd.cut(d[0],5).value_counts().sort_index().index[0]
    f = ~(d[0].between(iv.left,iv.right).values)
    di = f.sum()
    f = np.reshape(np.broadcast_to(f, np.append(x_train.shape[1:4], x_train.shape[0])), x_train.shape)
    x_train = x_train[f].reshape(np.insert(x_train.shape[1:4], 0, di, axis=0))
    y_train = y_train[f[:,0,0,0]]
print('Number of training images (after augmentation, excl. outliers): '+str(x_train.shape[0]))

### SELECT MODEL TYPE!
reg_model = 'LIN'

if (reg_model == 'CNN')|(reg_model=='NN'):
    description = input('Add description to log name: ')
    logdir = "logs/scalars/" + datetime.now().strftime("%y%m%d-%H%M")+ ' '+ str(description)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,profile_batch=0)


if reg_model == 'CNN':
    drop = 0.1
    model = models.Sequential()
    model.add(convolutional.Convolution2D(4, 3, 3, input_shape=(x_train.shape[1:4]), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(core.Flatten())
    model.add(core.Dense(200))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(100))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(80))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(40))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(20))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(10))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(10))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(10))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(10))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    # With CNN (Convolution2D(4,4,3),MaxPooling2D(2,2),dense(50),dropout(0.4),dense(30),dropout(0.4) the RMSE is 0.7
if (reg_model=='NN')|(reg_model=='LIN'): #Flatten from 8D image to array
    x_train = x_train.reshape([x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]])
    x_test = x_test.reshape([x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]])
if (reg_model == 'NN'):
    #drop = 0.15
    drop = 0.5
    model = models.Sequential()
    model.add(core.Dense(8000, input_shape=[x_train.shape[1]]))
    model.add(ReLU())
    #model.add(core.Dropout(drop)) #Dropout not recommended for initial layer.
    model.add(core.Dense(4000))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(2000))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(1000))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(500))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(250))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(125))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(50))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(25))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(12))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(6))
    model.add(ReLU())
    model.add(core.Dropout(drop))
    model.add(core.Dense(3))
    model.add(ReLU())
    model.add(core.Dropout(drop))
if (reg_model == 'CNN')|(reg_model=='NN'):
    model.add(core.Dense(1,activation='linear'))
    #model.add(ReLU())
    #model.compile(optimizer=optimizers.SGD(), loss='mean_squared_logarithmic_error')
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mse','mae'])
    history = model.fit(x_train, y_train,validation_split=0.1, batch_size=32, epochs=5,
                        callbacks=[tensorboard_callback],verbose=True)
    model.save('temp_model')
    #score = model.evaluate(x=x_test, y=y_test, batch_size=30, verbose=1)
    pred = model.predict(x_test)
if (reg_model == 'LIN'):
    ### Testing different regression models from Scikit-Learn
    reg = tree.DecisionTreeRegressor(criterion='friedman_mse', max_features=5) # Fairly good results! RMSE 0.4
    #reg = MLPRegressor(random_state=1, max_iter=10000) #Bad results!
    #reg = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=4), n_estimators=300) # Bad
    #reg = linear_model.LinearRegression() # Useless
    #reg = svm.SVR() # Bad results!
    #reg = SGDRegressor() #Useless!
    #reg = linear_model.Lasso() #Bad results!
    #reg = linear_model.ElasticNet(random_state=0) #Bad!
    #reg = linear_model.Ridge(alpha=.5) #Good! RMSE: 0.38
    #reg = linear_model.PoissonRegressor()
    reg = reg.fit(x_train, y_train)
    pred = reg.predict(x_test)

loss = mean_squared_error(y_test, pred, squared=False)
print('RMSE: '+str(loss))
truth = np.array([])
prediction = np.array([])
number = np.array([])
for i in range(0, x_test.shape[0]):
    truth = np.append(truth, np.full(shape=x_test[i, :].flatten().shape, fill_value=y_test[i]))
    prediction = np.append(prediction, np.full(shape=x_test[i, :].flatten().shape, fill_value=pred[i]))
    number = np.append(number, np.full(shape=x_test[i, :].flatten().shape, fill_value=i))
if False: #CAREFUL: Very slow plotting time if a large sample size is used.
    df = pd.DataFrame(np.transpose([truth,prediction,x_test.flatten(),number]),
                      columns=['Truth','Prediction','Pixel values','Sample'])
    sns.pairplot(df.sample(n=100),hue='Sample')
elif True:
    if reg_model=='CNN':
        x_train_temp = x_train.reshape([x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3]])
        x_test_temp = x_test.reshape([x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3]])
    else:
        x_train_temp = x_train
        x_test_temp = x_test
    if pred.ndim > 1:
        pred_temp = pred[:,0]
    else:
        pred_temp = pred
    ax = sns.scatterplot(x=y_test, y=pred_temp, size=x_test_temp.mean(axis=1), hue=x_test_temp.std(axis=1))
    ax.set(ylabel="Prediction", xlabel="Truth")
    ax.legend(title='Size=Mean & Hue=Std', loc='center right')
    if standardize:
        ax.set_ylim(-1,1)
        ax.set_xlim(-1, 1)
    else:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
else:
    #g = sns.jointplot(hue=x_test[:,:,:,:].flatten(),x=truth,y=prediction)
    g = sns.jointplot(hue=x_test.flatten(), x=truth, y=prediction)
    ax = g.ax_joint
    ax.set(ylabel="Prediction", xlabel = "Truth")
    ax.legend(title='Pixel value')
    # Add jitter
    dots = ax.collections[-1]
    offsets = dots.get_offsets()
    jittered_offsets = offsets + np.random.uniform(0, 1, offsets.shape)
    dots.set_offsets(jittered_offsets)
if False & (reg_model=='CNN'): # Plot an example training image.
    image_no = 5
    fig, axes = plt.subplots(2, 4)
    for i in range(0, 8):
        sns.heatmap(ax=axes.flat[i], data=x_train[image_no, :, :, i])

plt.tight_layout()
tracker.stop()
print('Done and done!')


