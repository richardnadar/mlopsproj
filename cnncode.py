#!/usr/bin/env python
# coding: utf-8

# In[32]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras

# loading data
(X_train, y_train), (X_test, y_test)  = mnist.load_data()


# In[33]:


# Plot images
from matplotlib import pyplot

# create a grid of 3x3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
    
# show the plot
pyplot.show()


# In[34]:


# Lets store the number of rows and columns
img_rows = X_train[0].shape[0]
img_cols = X_train[1].shape[0]

# Converting from 3-D to 4-D for keras
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# Change our image type to float32 data type
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
X_train /= 255
X_test /= 255

# One hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = X_train.shape[1] * X_train.shape[2]

# Data augmentation for more inputs
train_datagen = ImageDataGenerator(
                rotation_range = 30,
                zoom_range = 0.20, 
                fill_mode = "nearest", 
                shear_range = 0.20, 
                horizontal_flip = True, 
                width_shift_range = 0.1,
                height_shift_range = 0.1)

for X_batch, y_batch in train_datagen.flow(X_train, y_train, batch_size=9):
    # create a grid of 3x3 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    # show the plot
    pyplot.show()
    break


# In[35]:


# model creation
model = Sequential()

# 1 set of CRP (Convolution, RELU, Pooling)
def addlayers(i):
    model.add(Convolution2D(20*i, 
                            kernel_size = (3, 3),
                            padding = "same", 
                            input_shape = input_shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))


# In[36]:


addlayers(2)


# In[37]:


addlayers(4)
addlayers(6)


# In[18]:





# In[38]:


model.summary()


# In[39]:


# Fully connected layers (w/ RELU)
model.add(Flatten())
model.add(Dense(300))
model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))
           
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'Adam',
              metrics = ['accuracy'])
    
print(model.summary())


# In[40]:


# Training our model
result = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size = 32),
                                      validation_data = (X_test, y_test), 
                                      steps_per_epoch = len(X_train) // 32,
                                      validation_steps= len(X_test) // 32,
                                      epochs = 5)

# Evaluate the performance of our trained model
result.history['val_accuracy'][4]


# In[ ]:





# In[42]:


file1=open("accuracy.txt","w")
file1.write(str(result.history['val_accuracy'][4]*100))
file1.close()


# In[41]:


model.save('cnn_model.h5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




