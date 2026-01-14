import os
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
os.environ['CUDA_VISIBLE_DEVICES']: -1

class YOLODataReader(CalibrationDataReader):
    def __init__(self, model_path, batch_size=1):
        session = ort.InferenceSession(model_path)
        self.input_name = session.get_inputs()[0].name
        self.data_list = [np.random.rand(1, 3, 640, 640).astype(np.float32) for _ in range(10)]
        self.enum_data = iter([{self.input_name: d} for d in self.data_list])

    def get_next(self):
        return next(self.enum_data, None)

def export_all_formats(model_path):
    model = YOLO(model_path)
    base_name = "yolo11n_seg_quantized"

    print("--- 1/3 Export FP32 ---")
    if not os.path.exists(f"{base_name}_fp32.onnx"):
        path_fp32 = model.export(format="onnx", imgsz=640, simplify=True)
        os.rename(path_fp32, f"../models/{base_name}_fp32.onnx")

    print("\n--- 2/3 Export FP16 ---")
    if not os.path.exists(f"{base_name}_fp16.onnx"):
        path_fp32 = model.export(format="onnx", imgsz=640, simplify=True, half=True, device=0)
        os.rename(path_fp32, f"../models/{base_name}_fp16.onnx")

    print("\n--- 3/3 Quantification INT8 Statique ---")
    if not os.path.exists(f"{base_name}_int8.onnx"):
        dr = YOLODataReader(f"../models/{base_name}_fp32.onnx")
        quantize_static(
            model_input=f"../models/{base_name}_fp32.onnx",
            model_output=f"../models/{base_name}_int8.onnx",
            calibration_data_reader=dr,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8
        )

if __name__ == "__main__":
    export_all_formats('yolo11n-seg.pt')