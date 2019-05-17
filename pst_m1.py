from __future__ import absolute_import, division, print_function, unicode_literals
from time import time
import tensorflow as tf
from tensorflow import math
import numpy as np
from PST_func import PST

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#data prep
BATCH_SIZE = 1
SEED = 2

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[... , tf.newaxis]
x_test = x_test[... , tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

#Setting metics and training methods
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

EPOCHS = 10
template = "Epoch {}: Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}\n"

class pst_basic(tf.keras.layers.Layer):
    def __init__(self):
        super(pst_basic, self).__init__()

    def build(self, input_shape):
        self.pst_train = self.add_variable(name="pst_basic_trainable",
                                        shape=[5,1],
                                        dtype=tf.dtypes.float32,
                                        trainable=True,
                                        initializer=tf.initializers.TruncatedNormal(0.50, 0.25, seed=SEED),
                                        constraint=lambda var: tf.clip_by_value(var, 0, 1))
        self.pst_static = self.add_variable(name="pst_basic_satic",
                                        shape=[1,1],
                                        dtype=tf.dtypes.float32,
                                        initializer=tf.constant_initializer(0),
                                        trainable=False,
                                        constraint=lambda var: tf.clip_by_value(var, 0, 1))

    def call(self, input):
        return(PST(input,
        self.pst_train[0],
        self.pst_train[1],
        self.pst_train[2],
        self.pst_train[3],
        self.pst_train[4],
        self.pst_static[0]))


class pstModel(Model):
  def __init__(self):
    super(pstModel, self).__init__()
    self.pst1 = pst_basic()
    #Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.pst1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

#the train/test functions
model = pstModel()
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

  @tf.function
  def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

#the main train/test loop'
'''
model.build((32,28,28,1))
model.summary()
raise
'''

for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}: training....")

    start = time()
    for images, labels in train_ds:
        train_step(images, labels)
    end = time()
    print(f"\tTraining completed in:{end-start} sec")

    start = time()
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    end = time()
    print(f"\tTesting completed in:{end-start} sec")

    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
