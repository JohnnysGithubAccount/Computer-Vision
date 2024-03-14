az_path = r"D:\UsingSpace\Projects\Artificial Intelligent\Datasets\A_Z Handwritten Data\A_Z Handwritten Data.csv"
model_path = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision\OpticalCharacterRecognition\BasicWithTensorflowKeras\TrainedModel"
plot_path = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision\OpticalCharacterRecognition\BasicWithTensorflowKeras\Plot"
weight_path = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision\OpticalCharacterRecognition\BasicWithTensorflowKeras\BestWeights"
image_path = r"D:\UsingSpace\Projects\Artificial Intelligent\ComputerVision\OpticalCharacterRecognition\BasicWithTensorflowKeras\Images\image1.png"

# weight_file_name = r"\best_weight_333_64_0.h5"
# model_name_file = r"\model_333_64_0.h5"
weight_file_name = r"\best_weight_3333_64_512.h5"
model_name_file = r"\model_3333_64_512.h5"
plot_file_name = r"\plot_3333_64_512.png"
epochs = 100
init_lr = 1e-1
batch_size = 128
dropout_rate=0.5
reg = 0.0005
num_classes = 36
input_shape = (32, 32, 1)