import torch
from onnx2torch import convert

# Path to your ONNX model
onnx_model_path = "/home/andrus/.cache/huggingface/hub/models--louisJLN--yolo8-fashionpedia/snapshots/f98e49e0336097c355473cbb85e8187770820521/results/yolov8s-fashionpedia-1.onnx"

# Convert the model to PyTorch
torch_model = convert(onnx_model_path)

# Save as a .pt file
torch.save(torch_model, "model.pt")