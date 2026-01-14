from pathlib import Path

from .backends import OakDBackend
from .pipeline import benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOB_MODELS = PROJECT_ROOT / "models" / "blob"
OAK_CONFIG = Path(__file__).parent / "helpers" / "config.json"

models_files = {}

for model_path in BLOB_MODELS.iterdir():
    if model_path.suffix == ".blob":
        models_files[model_path.name] = model_path
        print(f"🔹 Modèle trouvé : {model_path.name}")

benchmark.bench(
    OakDBackend, 
    models_files, 
    PROJECT_ROOT / "results" / "oak_d", 
    backend_kwargs={"config_path": OAK_CONFIG}
)