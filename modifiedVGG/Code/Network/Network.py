"""

Author(s):
Kumar Gaurav
Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
#from tf.keras.layers import Input, tf.keras.layers.add, tf.keras.layers.Dense, tf.keras.layers.Activation, tf.keras.layers.ZeroPadding2D, tf.keras.layers.BatchNormalization, tf.keras.layers.Flatten, tf.keras.layers.Conv2D, AveragePooling2D, MaxPooling2D
#from keras.initializers import glorot_uniform

# Don't generate pyc codes
sys.dont_write_bytecode = True


def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    cnet1=conv_net(Img,32)
    #cnet2=tf.layers.tf.keras.layers.Conv2D(cnet1,64,(3,3),tf.keras.layers.Activation='relu',ptf.keras.layers.adding='same')
    #cnet2=tf.layers.max_pooling2d(cnet2,2,2)
    cnet2=conv_net(cnet1,64)
    cnet3=conv_net(cnet2,128)
    
    flat= tf.layers.Flatten(cnet3)
    prLogits=tf.layers.Dense(flat,128, Activation='relu',kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.001))
    #prLogits1=tf.layers.tf.keras.layers.Dense(prLogits2,32,tf.keras.layers.Activation='relu',use_bias=True)
    #prLogits=tf.layers.tf.keras.layers.Dense(prLogits1,10)
    prSoftMax=tf.layers.Dense(prLogits,10, Activation='softmax')

    return prLogits, prSoftMax

def conv_net(x, s):
    # tf.keras.layers.Conv2D wrapper, with bias and relu tf.keras.layers.Activation
    conv1=tf.layers.Conv2D(x,s,(3,3),Activation='relu',use_bias=True,padding='same',kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.001))
    conv2=tf.layers.Conv2D(conv1,s,(3,3),Activation='relu',use_bias=True,padding='same',kernel_initializer='he_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.001))
    #conv1=tf.layers.tf.keras.layers.Conv2D(x,s,(3,3),tf.keras.layers.Activation='relu',ptf.keras.layers.adding='same')
    #conv1=tf.nn.relu(conv1)
    #conv1=tf.layers.max_pooling2d(conv1,2,2)
    #conv2=tf.layers.tf.keras.layers.Conv2D(conv1,(2*s),(3,3),tf.keras.layers.Activation='relu',ptf.keras.layers.adding='same')
    conv2=tf.layers.batch_normalization(conv2)
    conv2=tf.layers.max_pooling2d(conv2,2,2)
    conv2=tf.layers.dropout(conv2,0.2)
    return conv2

def ResNet50(input1, input_shape, classes ):   
    # Define the input as a tensor with shape input_shape
    X_input = input1

    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = tf.keras.layers.Conv2D(32, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')
    X = identity_block(X, 3,  [32, 32, 128], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=5, block='c')

    # AVGPOOL.
    X = tf.layers.average_pooling2d(X,2,strides=(2, 2), padding='same')

    # output layer
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    
    # Create model
    #model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = tf.keras.layers.Conv2D(F1, (1, 1), strides = (s,s),padding='valid', name = conv_name_base + '2a', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    
    ##### SHORTCUT PATH ####
    X_shortcut = tf.keras.layers.Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1',padding='same', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: tf.keras.layers.add shortcut value to main path, and pass it through a RELU tf.keras.layers.Activation
    X = tf.keras.layers.add([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)
    
    return X

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. We'll need this later to tf.keras.layers.add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: tf.keras.layers.add shortcut value to main path, and pass it through a RELU tf.keras.layers.Activation
    X = tf.keras.layers.add([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


