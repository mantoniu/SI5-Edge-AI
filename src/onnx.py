from pathlib import Path

from .backends import OnnxBackend
from .pipeline import benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_MODELS = PROJECT_ROOT / "models"

models_files = {}

for model_path in ONNX_MODELS.iterdir():
    if model_path.suffix == ".onnx":
        models_files[model_path.name] = model_path
        print(f"🔹 Modèle trouvé : {model_path.name}")

benchmark.bench(
    OnnxBackend, 
    models_files,
    PROJECT_ROOT / "results",
    result_path = 'benchmark_onnx.csv'
)