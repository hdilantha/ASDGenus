import sys
import mne
import csv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path = "D:/FYP/implementation/EEG Data/EEG Temp 002/ADOS_Mod3002.vhdr"
csv_file = 'C:/Users/User/Documents/NetBeansProjects/ASDGenus/src/mn_std_ent_9_6_unscaled_cfs.csv'

def build_model(train_data):
  model = keras.Sequential([
          layers.Flatten(input_shape=[train_data.shape[1]]),
          layers.Dense(128, activation = tf.nn.relu),
          layers.Dense(4, activation = tf.nn.softmax)
  ])

  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss = 'mean_squared_error', optimizer = optimizer)
  return model

raw = mne.io.read_raw_brainvision(path, preload=True) # sys.argv[1]

raw.pick_types(meg=False, eeg=True, eog=True)

raw_avg_ref, _ = mne.set_eeg_reference(raw, ref_channels='average')
raw_avg_df = raw_avg_ref.to_data_frame()
raw_avg_arr = raw_avg_df.to_numpy()

file_name = path.split('/')[-1].split('.')[0] # sys.argv[1]
features = []
   
mean_chs = [3, 13, 16, 18, 19, 25]
std_chs = [24, 28]
ent_chs = [14]

for ch in mean_chs:
    features.append(np.mean(raw_avg_arr[:, ch]))
for ch in std_chs:
    features.append(np.std(raw_avg_arr[:, ch]))
for ch in ent_chs:
    pArr = raw_avg_arr[:, ch] / sum(raw_avg_arr[:, ch])
    features.append(-np.nansum(pArr * np.log2(pArr)))
      
X = [features]
y = []
file_names = []

with open(csv_file, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    next(csvfile, None)
    for row in csvreader:
        if(row[0] != file_name):
            file_names.append(row[0])
            X.append([float(x) for x in row[1: len(row) -1]])
            y.append(row[-1])
            
X = np.array(X)
for i in range(X.shape[1]):
    X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
     
    
X_train = np.array(X[1:])
y_temp = []
for y_val in y:
    if y_val == 'n':
        y_temp.append([1., 0., 0., 0.])
    if y_val == 'p':
        y_temp.append([0., 1., 0., 0.])
    if y_val == 'l':
        y_temp.append([0., 0., 1., 0.])
    if y_val == 'h':
        y_temp.append([0., 0., 0., 1.])
    
y_train = np.array(y_temp)
X_test = np.array([X[0]])
    
model = build_model(X_train)

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30)
model.fit(X_train, y_train, epochs = 2000, validation_split = 0.1, verbose = 0, callbacks = [early_stop])

predictions = model.predict(X_test).flatten()

output = ''
predicted_level = int(np.where(predictions == max(predictions))[0])
if(predicted_level == 0):
    output = 'no ASD , Predicted ADOS-2 score range: 0 - 2' 
if(predicted_level == 1):
    output = 'potential ASD , Predicted ADOS-2 score range: 3 - 6'
if(predicted_level == 2):
    output = 'low ASD , Predicted ADOS-2 score range: 7 - 12'
if(predicted_level == 3):
    output = 'high ASD , Predicted ADOS-2 score range: 14 - 26'

print('Prediction: ', output)
