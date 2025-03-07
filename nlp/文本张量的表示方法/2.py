import tensorflow as tf
print("TensorFlow版本:", tf.version.VERSION)  # 使用更可靠的版本获取方式
print("GPU支持:", tf.test.is_gpu_available())  # 验证安装完整性