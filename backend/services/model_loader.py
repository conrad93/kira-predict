from joblib import load

_model_cache = {}

MODEL_PATHS = {
    "0": "models/model.joblib",
    "1": "models/model_real.joblib",
    "2": "models/model_v2.joblib"
}

def get_model(version):
    if version not in MODEL_PATHS:
        return None

    if version not in _model_cache:
        try:
            _model_cache[version] = load(MODEL_PATHS[version])
        except Exception as e:
            print(f"❌ Failed to load model {version}: {e}")
            return None

    return _model_cache[version]