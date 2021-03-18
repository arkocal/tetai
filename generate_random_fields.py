import random

import utils

for i in range(100_000):
    field = "".join(["".join([random.choice(["0", "1"]) for _ in range(10)])+"0"*12 for _ in range(10)])
    print(field)
