DEFAULT_DEVICE = "cpu"

SUPPORTED_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "int8",
    "int16",
    "int32",
}


def get_device(device: str | None = None) -> str:
    if device is None or device == "":
        return DEFAULT_DEVICE
    return device


def get_dtype(dtype: str | None = None) -> str:
    if dtype is None or dtype == "":
        return "float32"
    return dtype


def validate_dtype(dtype: str) -> bool:
    return dtype in SUPPORTED_DTYPES


def normalize_device_string(device: str) -> str:
    return device.lower().strip()
