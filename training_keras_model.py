####
#this program only runs after the hdf5 file is already stored in your computer
#best to run the training_scikit.py file first
###

import numpy as np
from spectral_tiffs import read_mtiff, read_stiff
import glob
import tensorflow as tf
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
#compile mask images
class_ids = { 'Blue dye' : 1,
              'Red dye': 2,
              'ICG': 3,
              'Stroma, ICG': 4,
              'Stroma': 5,
              'Umbilical cord': 6,
              'Suture': 7,
              'Artery, ICG': 8,
              'Vein': 9,
              'Artery': 10,
              }
train_masks = []
names = []
lastn = []
for img in glob.glob("OneDrive_1_3-13-2021/Set #1 Mask images/*"):
    names.append(img)

for name in sorted(names):
  namex = name.split("/")
  last = namex[-1].split(".")
  lastn.append(int(last[0]))

for msk_img in sorted(lastn):
    mask = read_mtiff("OneDrive_1_3-13-2021/Set #1 Mask images/"+str(msk_img)+".tif")
    temp = np.zeros((1024,1024),dtype = int)
    for key in mask:
      if key in class_ids:
        mask_temp = mask[key]*class_ids[key]

        temp = np.maximum(temp,mask_temp)
    train_masks.append(temp.astype(int))

train_masks = np.array(train_masks, dtype=np.float32)

#transform and store training images in HDF5 file
count_ignored = 0
count_ignored1 = 0
count_for_mask = 0
dataset_file = h5py.File('OneDrive_1_3-13-2021/dataset_allpixs.hdf5','a')



for img in glob.glob("OneDrive_1_3-13-2021/Set #1 Reflectance spectral images, 510-700 nm range/*"):
    spim,w,rgb,meta = read_stiff(img)

    sx = []
    for i in range (0,1024):
      for j in range (0,1024):
        temp = []
        for z in range (0,38):
          temp.append(spim[i,j,z])
        sx.append(temp)

    c = 0
    for i in range (0,1024):
      for j in range (0,1024):
        sx[c].append(int(train_masks[count_for_mask][i][j]))
        c = c+1
    count_for_mask=count_for_mask+1

    sx_array = np.array(sx)
    print(sx_array.shape)
    if count_for_mask == 1:
      dataset_file.create_dataset("dset_train", data=sx_array, maxshape=(None, 39))
    else:
      dataset_file['dset_train'].resize((dataset_file['dset_train'].shape[0] + sx_array.shape[0]),axis = 0)
      dataset_file['dset_train'][-sx_array.shape[0]:] = sx_array

dataset_file.close()

#read dataset
data= h5py.File('OneDrive_1_3-13-2021/dataset_allpixs.hdf5','r')
n1 = data.get('dset_train')

#select samples from dataset and store them in another dataset
dataset_file = h5py.File('OneDrive_1_3-13-2021/dataset_selected.hdf5','a')
limit = 100000
label= np.zeros((11))
i = 0
while (i<n1.shape[0] and (label[0] <= limit or label[1] <= limit or label[2] <= limit or label[3] <= limit or label[4] <= limit 
                          or label[5] <= limit or label[6] <= limit or label[7] <= 97355 or label[8] <= limit or label[9] <= limit or label[10] <= limit)):
  print(label)
  if label[int(n1[i,-1])] <= limit:
     label[int(n1[i,-1])] += 1
     temp_row = n1[i].reshape((1,39))
     if i == 0:
      dataset_file.create_dataset("dset_train", data=temp_row, maxshape=(None,39))
     else:
      dataset_file['dset_train'].resize((dataset_file['dset_train'].shape[0] + 1),axis = 0)
      dataset_file['dset_train'][-1:] = temp_row
  i=i+1
dataset_file.close()

selected_data= h5py.File('OneDrive_1_3-13-2021/dataset_selected.hdf5','r')
n1 = selected_data.get('dset_train')

n2 = np.array((n1.shape[0],n1.shape[1]))
n2 = np.array(n1).copy()
np.random.shuffle(n2)
X = n2[:,:-1]  
y = n2[:,-1]
X = np.array(X)
y = tf.keras.utils.to_categorical(y,num_classes=11)
# # Create the model
model = Sequential()
model.add(Dense(512, input_dim=(38), activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(11, activation='softmax'))



# # Compile the model
checkpnt_pth = 'OneDrive_1_3-13-2021/training/cp.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpnt_pth, save_weights_only = True, verbose=1)
opt = SGD(lr=0.01, momentum=0.9, decay=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#y_test = tf.keras.utils.to_categorical(y_test,num_classes=11)


history = model.fit(X,y,epochs=30, 
                    callbacks=[cp_callback],
                    validation_split = 0.2,
                    shuffle = True)

model.save("OneDrive_1_3-13-2021/model_keras.h5")
##### to show the order the images are read because the order proved to be not the
#### same as in the folder
#for img in glob.glob("/content/drive/MyDrive/OneDrive_1_3-13-2021/Set #1 Reflectance spectral images, 510-700 nm range/*"):
#  print(img)

#for img in glob.glob("/content/drive/MyDrive/OneDrive_1_3-13-2021/Set #1 Mask images/*"):
#  print(img)