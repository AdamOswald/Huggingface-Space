from io import BytesIO
from PIL import Image


def convert_bytes_to_pil(bytes_string: bytes) -> Image:
    buffer = BytesIO(bytes_string)
    buffer.seek(0)
    return Image.open(buffer)


def is_google_colab():
    try:
        import google.colab
            
        return True
    except:
        return False
