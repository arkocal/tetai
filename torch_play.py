import torch

from common import feature_vector
from torch_ai import DeepBoy, FEATURE_VEC_LEN, MODEL_PATH

model = DeepBoy(FEATURE_VEC_LEN)
model.load_state_dict(torch.load(MODEL_PATH))

field = [[False for _ in range(22)] for _ in range(10)]

def rate_field(field):
    feature_vec = torch.tensor(feature_vector(field), dtype=torch.float)
    with torch.no_grad():
        model_score = model(feature_vec)
        return model_score.item()

rate_field(field)
