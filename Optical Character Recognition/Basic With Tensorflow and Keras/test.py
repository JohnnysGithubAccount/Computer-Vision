import tensorflow as tf

model = tf.keras.models.load_model(r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision\OpticalCharacterRecognition\BasicWithTensorflowKeras\TrainedModel\model_3333_64_512.h5")

model.summary()