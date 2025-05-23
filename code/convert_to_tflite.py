import torch
from torchvision import models
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from pathlib import Path

def build_model(num_classes, checkpoint_path, device):
    # 1. Recreate the same PyTorch model architecture
    model = models.mobilenet_v3_small(pretrained=False)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)
    # 2. Load your best checkpoint
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def export_to_onnx(model, onnx_path, device):
    # Create a dummy input and export
    dummy = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        opset_version=11
    )
    print(f"Exported ONNX model to {onnx_path}")

def onnx_to_saved_model(onnx_path, saved_model_dir):
    # Load ONNX and convert to TensorFlow SavedModel
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(saved_model_dir))
    print(f"Exported TensorFlow SavedModel to {saved_model_dir}")

def saved_model_to_tflite(saved_model_dir, tflite_path):
    # Convert SavedModel to TFLite with optimizations
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Exported TFLite model to {tflite_path}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    ckpt_path    = project_root / 'models' / 'mobilenet_food101.pth'
    onnx_path    = project_root / 'models' / 'mobilenet_food101.onnx'
    saved_dir    = project_root / 'models' / 'mobilenet_food101_saved'
    tflite_path  = project_root / 'models' / 'mobilenet_food101.tflite'

    device = torch.device('cpu')  # ONNX export uses CPU
    model = build_model(num_classes=101, checkpoint_path=ckpt_path, device=device)

    export_to_onnx(model, onnx_path, device)
    onnx_to_saved_model(onnx_path, saved_dir)
    saved_model_to_tflite(saved_dir, tflite_path)
