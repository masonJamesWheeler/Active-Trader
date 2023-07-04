from collections import deque
import numpy as np
import pandas as pd
deq = deque(maxlen=128)
# loop 128 times
for i in range(128):
    # create random array shaped (1, 32)
    new_data = np.random.rand(32)
    # append to deque
    deq.append(new_data)

# convert deque to array
data = np.array(deq)
print(data.shape)
