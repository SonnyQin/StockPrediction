import tensorflow as tf


# 设计网络结构
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(28),
        tf.keras.layers.LSTM(28),
        tf.keras.layers.Dense(10,activation="sigmoid")
        ],name="LSTM")

# 定义代价函数
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 模型编译
model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])