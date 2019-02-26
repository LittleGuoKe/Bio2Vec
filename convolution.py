'''
1.Loading data and then divide it into x_train, y_train,x_test, y_test.
2.Forward propagation.
3.Back propagation.
4.Training the network
'''
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Input, Dense, Flatten,  Dropout,concatenate
from keras.layers.advanced_activations import PReLU
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping
from util import make_data
# from loading_data import load_data
from keras.models import Model
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from keras.utils import plot_model
import datetime

print("start",datetime.datetime.now())
# define parameters
patience=50
# model_save_path='model/'+"patience_"+str(patience)+"_human_windows_size_3_model"
model_save_path='model/'+"patience_"+str(patience)+"_human_test_2048dim_20000unigram"
'''data'''
max_feature =4096
batch_size = 64
'''convolution layer'''
filters = 64
kernel_size = 2
pool_size = 2
strides = 1
log_dir = 'log_dir/'
acc_loss_file = 'images/human2_concatenate_cnn_acc_loss_sg__human_test_2048dim_20000unigram'
# num_classes = 2
epochs =50

# get data
x, y = make_data()

# expected input data shape: (batch_size, timesteps, data_dim)
x = x.reshape(-1, 1, max_feature)
# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=64)

# init = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=64)

digit_input = Input(shape=(1, max_feature))
x = Conv1D(filters, kernel_size, padding="same")(digit_input)
x = Conv1D(filters, kernel_size, padding="same", )(x)
x = Dropout(0.5)(x)
x=PReLU()(x)
x = MaxPooling1D(pool_size=pool_size, strides=strides, padding="same")(x)
y=Conv1D(32, kernel_size, padding="same")(digit_input)
y = Dropout(0.5)(y)
y=PReLU()(y)
y = MaxPooling1D(pool_size=pool_size, strides=strides, padding="same")(y)
k=Conv1D(128, kernel_size, padding="same")(digit_input)
k = Dropout(0.5)(k)
k=PReLU()(k)
k= MaxPooling1D(pool_size=pool_size, strides=strides, padding="same")(k)
z=concatenate([x,y,k])
z = Flatten()(z)

# x = Flatten()(x)
# x = GRU(gru_output_size, dropout=0.5, recurrent_dropout=0.5)(x)
out = Dense(1, activation='sigmoid')(z)
model = Model(digit_input, out)
model.summary()

print('Compiling the Model...')
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

print("Train...")

# store checkpoint file every 5 steps
# checkpointer = ModelCheckpoint(filepath="model/"+"patience_"+str(patience)+"human_test_64dim_20000unigram_"+"weights.{epoch:02d}.hdf5", verbose=1, period=10)
#
# TB = TensorBoard(log_dir=log_dir, write_images=1, histogram_freq=1, write_graph=True)
early_stopping = EarlyStopping(monitor='val_loss',patience=patience)

fit_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True,
                        callbacks=[early_stopping])

'''save final model'''
model.save(model_save_path)

'''plot acc and loss'''
plt.figure(1)
plt.subplot(1, 2, 1)
x = range(1, len(fit_history.history['loss']) + 1)
plt.plot(x, fit_history.history['loss'], 'b', x, fit_history.history['val_loss'], 'r')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(x, fit_history.history['acc'], 'b', x, fit_history.history['val_acc'], 'r')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(acc_loss_file)

print("Evaluate...")
test_history = model.evaluate(x_test, y_test, batch_size=batch_size)
print('patience: ',patience)
print(model.metrics_names)
print('Test score:', test_history)
print("end",datetime.datetime.now())
# '''
# store Neural Network,include: both graph and weight
# '''
# model.save('model/NN_model.h5')

'''
visualization Neural Network
'''
# plot_model(model, to_file='images/lstm_model.png', show_shapes=True)


'''
store accuracy on matine_Final set and fit_history
'''

# train_acc = fit_history.history['acc']
# train_loss = fit_history.history['loss']
# val_acc = fit_history.history['val_acc']
# val_loss = fit_history.history['val_loss']
# epochs = fit_history.epoch
#
# with open('data/test_set_accuracy.pickle1', mode='wb') as f:
#     pickle.dump([test_history, train_acc, train_loss, val_acc, val_loss, epochs], f, -1)
