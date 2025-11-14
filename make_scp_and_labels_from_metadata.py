import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/SincNet/flora_voice_dataset/metadata.csv")
emotions = sorted(df["emotion"].unique())

emotion_id = {}
for i, e in enumerate(emotions):
    emotion_id[e] = i

os.makedirs("data_lists", exist_ok=True)

# Create dev split if missing
if "dev" not in df["split"].unique():
    train_df = df[df["split"] == "train"]
    train, dev = train_test_split(train_df, test_size=0.1, stratify=train_df["emotion"], random_state=42)
    test = df[df["split"] == "test"]
    train["split"], dev["split"] = "train", "dev"
    df = pd.concat([train, dev, test])

# --- Write SCP files using itertuples() ---
def write_scp(df_subset, fname):
    with open(fname, "w") as f:
        for row in df_subset.itertuples(index=False):
            f.write(os.path.join(row.emotion, row.file_name) + "\n")

write_scp(df[df.split == "train"], "data_lists/EMO_train.scp")
write_scp(df[df.split == "dev"], "data_lists/EMO_dev.scp")
write_scp(df[df.split == "test"], "data_lists/EMO_test.scp")

# --- Write labels using itertuples() ---
labels = {
    os.path.join(row.emotion, row.file_name): emotion_id[row.emotion]
    for row in df.itertuples(index=False)
}
np.save("data_lists/EMO_labels.npy", labels)

print("SCP + label files created successfully.")
