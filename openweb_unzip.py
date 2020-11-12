import os
import tqdm

input_dir = "/cfs/aidenliang/corpus/openwebtxt"
for parent, dirnames, filenames in os.walk(input_dir):
    filenames = list(filenames)
    for filename in tqdm.tqdm(filenames):
        os.system(f"xz -d {filename}")
