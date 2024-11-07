import onnxruntime as rt

from pdftext.settings import settings


def get_model(model_path=settings.MODEL_PATH):
    sess = rt.InferenceSession(model_path)
    return sess
