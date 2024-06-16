import numpy as np
import os 
save_dir = "plots"
TEST_NOS = 5
seeds = np.random.randint(0, 100, TEST_NOS)
print("Seeds: ", seeds)

from module import build_and_run
from tqdm import tqdm
sum = 0

for filename in os.listdir(save_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        os.remove(os.path.join(save_dir, filename))  
        
for i in tqdm(range(TEST_NOS)):
    seed = seeds[i]
    sum += build_and_run(seed, True, False)
    tqdm.write(f"Times won:{sum}")
print(f"success rate = {sum/TEST_NOS} ")

sum = 0
for i in tqdm(range(TEST_NOS)):
    seed = seeds[i]
    sum += build_and_run(seed, False, False)
    tqdm.write(f"Times won:{sum}")
print(f"success rate = {sum/TEST_NOS} ")
