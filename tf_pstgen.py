from __future__ import absolute_import, division, print_function, unicode_literals
from time import time
import tensorflow as tf
import numpy as np
from PST_func import PST

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[... , tf.newaxis]
x_test = x_test[... , tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(1)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

class basicModel(Model):
  def __init__(self):
    super(basicModel, self).__init__()
    #self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    #x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = basicModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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


EPOCHS = 10
template = "Epoch {}: Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}\n"

for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}: training....")

    start = time()
    for images, labels in train_ds:
        #print('{}'.format(images.shape))
        images = np.reshape(images,(28,28))
        [out, kernel] = PST(images,0.5,0.2,9.6,-0.5,0.01,1)
        images = np.reshape(np.array(out), (1,28,28,1))
        train_step(images, labels)
    end = time()
    print(f"\tTraining completed in:{end-start} sec")

    start = time()
    for test_images, test_labels in test_ds:
        images = np.reshape(test_images,(28,28))
        [out, kernel] = PST(images,0.5,0.2,9.6,-0.5,0.01,1)
        test_images = np.reshape(np.array(out), (1,28,28,1))
        test_step(test_images, test_labels)
    end = time()
    print(f"\tTesting completed in:{end-start} sec")

    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
