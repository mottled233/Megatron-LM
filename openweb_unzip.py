import os
import tqdm

input_dir = "/cfs/aidenliang/corpus/openwebtext"
for parent, dirnames, filenames in os.walk(input_dir):
    filenames = list(filenames)
    for filename in tqdm.tqdm(filenames):
        current = os.path.join(parent, filename)
        os.system(f"xz -d {current}")
