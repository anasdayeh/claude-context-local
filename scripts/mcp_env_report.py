import os
import platform
import sys
import sysconfig

print("PYTHON:", sys.version)
print("EXECUTABLE:", sys.executable)
print("PLATFORM:", platform.platform())
print("MACHINE:", platform.machine())
print("GIL_DISABLED:", getattr(sys, "_is_gil_enabled", lambda: "n/a")() is False)
print("CONFIG_ARGS:", sysconfig.get_config_var("CONFIG_ARGS"))

for key in [
    "CODE_SEARCH_DEVICE",
    "CODE_SEARCH_EMBED_BACKEND",
    "CODE_SEARCH_INDEX_WORKERS",
    "CODE_SEARCH_EMBED_BATCH_SIZE",
    "CODE_SEARCH_CHUNK_BATCH_SIZE",
    "CODE_SEARCH_PDF_OCR",
    "CODE_SEARCH_PDF_OCR_LANGUAGES",
    "CODE_SEARCH_PDF_OCR_MIN_TEXT_CHARS",
    "CODE_SEARCH_TORCH_NUM_THREADS",
    "CODE_SEARCH_TORCH_INTEROP_THREADS",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    print(f"{key}={os.getenv(key)}")
