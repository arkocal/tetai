import os

os.system(f"shuf scores/score_rulebased -n 5000 > /tmp/scores && python torch_ai.py --dump models/experimental/tensor_0")
exit()
for i in range(50):
    print(i)
    os.system(f"shuf scores/score_rulebased -n 5000 > /tmp/scores && python torch_ai.py --dump models/experimental/tensor_{i+1} --start_from models/experimental/tensor_{i}")
