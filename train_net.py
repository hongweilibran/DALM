import numpy as np
import os
from model import *
from Statistics import *
#from dataProc import *
from loadData_all import load_data
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

lr = 2e-4
numClass = 5
model_path = 'saved_models'
model_name = 'HyperM-Net'
train_list = ['7', '8_2', '9_4', '14', '20_2', '20', '23', '30']
val_list = ['8', '10_2', '10', '22_1', '31']

#load and generate training data.
X_train, y_train, X_val, y_val = load_data(train_list, val_list, augment=True)

model = D_Unet(numClass)
# model = Unet(numClass)
model.summary()
model.compile(optimizer=Adam(lr=lr), loss=Dice_loss, metrics=[Dice])
# callbacks = [
# tf.keras.callbacks.EarlyStopping(patience=2, monitor='loss')]
mdl_history = model.fit(X_train, y_train, batch_size=3, epochs=30)

model.save(os.path.join(model_path, 'dual_input.h5'))
pd.DataFrame(mdl_history.history).to_csv("results/"+model_name+".csv")

plt.figure()
plt.plot(mdl_history.history['train_loss'], 'b-', label='train')
#plt.plot(mdl_history.history['val_loss'], 'r-', label='val')
plt.legen()
plt.savefig('plots/loss_curves.png')


