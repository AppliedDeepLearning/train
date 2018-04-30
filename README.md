Training utilities for TensorFlow.


<!-- TOC depthFrom:2 depthTo:3 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Installation](#installation)
- [Getting Started](#getting-started)
	- [Model Function](#model-function)
	- [Training Mode](#training-mode)
- [License](#license)

<!-- /TOC -->


## Installation

```sh
pip install train
```

It is recommended to use a [virtual environment].


## Getting Started

```py
from train import Model, GradientDescent
import tensorflow as tf

# Define the network architecture - layers, number of units, activations etc.
def network(inputs):
    hidden = tf.layers.Dense(units=64, activation=tf.nn.relu)(inputs)
    outputs = tf.layers.Dense(units=10)(hidden)
    return outputs

# Configure the learning process - loss, optimizer, evaluation metrics etc.
model = Model(network,
              loss='sparse_softmax_cross_entropy',
              optimizer=GradientDescent(0.001),
              metrics=['accuracy'])

# Train the model using training data
model.train(x_train, y_train, epochs=30, batch_size=128)

# Evaluate the model performance on test or validation data
loss_and_metrics = model.evaluate(x_test, y_test)

# Use the model to make predictions for new data
predictions = model.predict(x)
# or call the model directly
predictions = model(x)
```

More configuration options are available:

```py
model = Model(network,
              loss='sparse_softmax_cross_entropy',
              optimizer=GradientDescent(0.001),
              metrics=['accuracy'],
              model_dir='/tmp/my_model')
```

You can also use custom functions for loss and metrics:

```py
def custom_loss(labels, outputs):
    pass

def custom_metric(labels, outputs):
    pass

model = Model(network,
              loss=custom_loss,
              optimizer=GradientDescent(0.001),
              metrics=['accuracy', custom_metric])
```

### Model Function

To have more control, you may configure the model inside a function using `Estimator` class:

```py
from train import Estimator, PREDICT
import tensorflow as tf

def model(features, labels, mode):
    # Define the network architecture
    hidden = tf.layers.Dense(units=64, activation=tf.nn.relu)(features)
    outputs = tf.layers.Dense(units=10)(hidden)
    predictions = tf.argmax(outputs, axis=1)
    # In prediction mode, simply return predictions without configuring learning process
    if mode == PREDICT:
        return predictions

    # Configure the learning process for training and evaluation modes
    loss = tf.losses.sparse_softmax_cross_entropy(labels, outputs)
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    accuracy = tf.metrics.accuracy(labels, predictions)
    return dict(loss=loss,
                optimizer=optimizer,
                metrics={'accuracy': accuracy})

# Create the model using model function
model = Estimator(model)

# Train the model
model.train(x_train, y_train, epochs=30, batch_size=128)
```

`mode` parameter specifies whether the model is used for training, evaluation or prediction.

### Training Mode

For layers like `Dropout`, you may use the training mode variable:

```py
from train import training

x = tf.layers.Dropout(rate=0.4)(x, training=training())
```

`Model` and `Estimator` classes automatically manage the training mode variable. To change it manually, use:

```py
# Enter training mode
training(True, session=session)

# Exit training mode
training(False, session=session)
```


## License

[MIT][license]


[license]: /LICENSE
[virtual environment]: https://docs.python.org/3/library/venv.html
