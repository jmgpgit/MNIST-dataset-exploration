import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # input layer, flatten 28x28 image to 784x1
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # hidden layer, 128 neurons
    tf.keras.layers.Dropout(0.2), # dropout layer, 20% of neurons are randomly dropped
    tf.keras.layers.Dense(10) # output layer, 10 neurons
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

def p_model(test): 
  return list(map(lambda x : np.argmax(x),probability_model.predict(test)))