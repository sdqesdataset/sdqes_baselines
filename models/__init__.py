from .encode_pool_classify import EncodePoolClassifyModel, EncodePoolClassifyStreamingModel
from .random import RandomModel
# from .text4vis import Text4VisModel
# from .st_adapter import STAdapterModel

def get_model_class(model_name):
    if model_name == "encode_pool_classify":
        return EncodePoolClassifyModel
    elif model_name == "encode_pool_classify_streaming":
        return EncodePoolClassifyStreamingModel
    elif model_name == "random":
        return RandomModel
    # elif model_name == "text4vis":
    #     return Text4VisModel
    # elif model_name == "st_adapter":
    #     return STAdapterModel

    raise ValueError(f"Unknown model name: {model_name}")
