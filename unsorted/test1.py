import onnx
import torch
from onnx2pytorch import ConvertModel

onnx_model_path = "/home/andrus/.cache/huggingface/hub/models--louisJLN--yolo8-fashionpedia/snapshots/f98e49e0336097c355473cbb85e8187770820521/results/yolov8s-fashionpedia-1.onnx"

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Convert to PyTorch
pytorch_model = ConvertModel(onnx_model)

print(dir(pytorch_model))
# Save the result
#pytorch_model.save("model2.pt")
# torch.save(pytorch_model, "model1.pt")
