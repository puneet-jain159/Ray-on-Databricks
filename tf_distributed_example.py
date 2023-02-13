# Databricks notebook source
# MAGIC %md 
# MAGIC # Runnig Distributed tensorflow with Ray
# MAGIC  
# MAGIC DBR version : **12.0 (ML) with GPU** </br>
# MAGIC Dataset : **FashionMNIST**

# COMMAND ----------

# MAGIC %md ## Initialize Ray cluster

# COMMAND ----------

# MAGIC %run ./initialize_ray $version="2.2.0"

# COMMAND ----------

from utils import get_dashboard_url
print(f"Link to ray Dashboard : {get_dashboard_url(spark,dbutils)}")

# COMMAND ----------

# MAGIC %md ### Import correct libraries

# COMMAND ----------

import json
import os

import numpy as np
from ray.air.result import Result
import tensorflow as tf

from ray.train.tensorflow import TensorflowTrainer
from ray.air.integrations.keras import Callback as TrainCheckpointReportCallback
from ray.air.config import ScalingConfig

# COMMAND ----------

# MAGIC %md ### Import the Data

# COMMAND ----------

# Download training data from open datasets.
def mnist_dataset(batch_size: int) -> tf.data.Dataset:
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(60000)
        .repeat()
        .batch(batch_size)
    )
    return train_dataset

# COMMAND ----------

# MAGIC  %md ### Define the Architecture

# COMMAND ----------

# Define model
def build_cnn_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    return model

# COMMAND ----------

def train_func(config: dict):
    per_worker_batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 3)
    steps_per_epoch = config.get("steps_per_epoch", 70)

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_cnn_model()
        learning_rate = config.get("lr", 0.001)
        multi_worker_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

    history = multi_worker_model.fit(
        multi_worker_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[TrainCheckpointReportCallback()],
    )
    results = history.history
    return results

# COMMAND ----------

ray.init(ignore_reinit_error=True )

# COMMAND ----------

def train_tensorflow_mnist(
    num_workers: int = 2, use_gpu: bool = False, epochs: int = 4
) -> Result:
    config = {"lr": 1e-3, "batch_size": 64, "epochs": epochs}
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=num_workers,
                                     resources_per_worker = {"CPU" : 3},
                                     use_gpu=use_gpu),
    )
    results = trainer.fit()
    return results


# COMMAND ----------

train_tensorflow_mnist(
  num_workers=1, use_gpu=True, epochs=25
)

# COMMAND ----------


