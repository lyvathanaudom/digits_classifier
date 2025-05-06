# temporary use
import os
import pandas as pd
from glob import glob

out = []
files = glob("data/Handwritten-Khmer-Digit/*/*/*.jpg")
for f in files:
    filename = f.split(os.sep)[-1]
    label = f.split(os.sep)[-2]
    print(filename, label)
    out.append({
        "filename": filename,
        "label": label
    })

df = pd.DataFrame(out)
df.to_csv("data\khmer-handwritten-digits\data.csv", index=False)