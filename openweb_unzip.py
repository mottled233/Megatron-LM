import os
import tqdm

input_dir = "/cfs/corpus/openwebtxt"
for parent, dirnames, filenames in os.walk(input_dir):
    filenames = list(filenames)
    print(len(filenames))