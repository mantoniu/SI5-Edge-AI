import os
from pathlib import Path
import onnx
from onnxconverter_common import float16
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

def _normalize_onnx_value_info(model_path):
    model = onnx.load(str(model_path))
    def _clear_value_info_in_graph(graph):
        del graph.value_info[:]
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _clear_value_info_in_graph(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        _clear_value_info_in_graph(subgraph)

    _clear_value_info_in_graph(model.graph)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, str(model_path))

def export_all_formats(model_path):
    model = YOLO(model_path)
    base_name = "yolo11n_seg"
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir.parents[1] / "models" / "onnx"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Export FP32")
    fp32_path = models_dir / f"{base_name}_fp32.onnx"
    path_fp32 = model.export(format="onnx", imgsz=640, simplify=True)
    os.replace(path_fp32, fp32_path)

    print("\nExport FP16")
    fp16_path = models_dir / f"{base_name}_fp16.onnx"
    model_fp16 = float16.convert_float_to_float16(
        onnx.load(str(fp32_path)),
        keep_io_types=True,
        op_block_list=["NonMaxSuppression", "Resize"]
    )
    onnx.save(model_fp16, str(fp16_path))
    _normalize_onnx_value_info(fp16_path)

    print("\nINT8 Dynamic Quantization")
    int8_path = models_dir / f"{base_name}_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

if __name__ == "__main__":
    export_all_formats("yolo11n-seg.pt")
