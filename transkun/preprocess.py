from pathlib import Path
import argparse
import random
import pandas as pd

TESTSET_RATIO = 0.1

DATASET_PATH = Path("../data")
EGDB_PATH = DATASET_PATH / "EGDB_flac"
GAPS_PATH = DATASET_PATH / "gaps_v1"
GUITARSET_PATH = DATASET_PATH / "guitarset_yourmt3_16k"

'''
# EGDB
labels = list(EGDB_PATH.glob("audio_label/*.midi"))
# select 10% of the labels
test_labels = random.sample(labels, int(len(labels) * TESTSET_RATIO))
test_id = [label.stem for label in test_labels]

train_set = []
test_set = []

for audio in list(EGDB_PATH.glob("audio*/*.flac")):
    if audio.stem in test_id:
        test_set.append((audio.relative_to(EGDB_PATH), f'audio_label/{audio.stem}.midi'))
    else:
        train_set.append((audio.relative_to(EGDB_PATH), f'audio_label/{audio.stem}.midi'))

# TODO add real data on test set

# save as csv
train_set_df = pd.DataFrame(train_set, columns=["audio", "label"])
test_set_df = pd.DataFrame(test_set, columns=["audio", "label"])
train_set_df.to_csv(DATASET_PATH / "EGDB_train.csv", index=False)
test_set_df.to_csv(DATASET_PATH / "EGDB_test.csv", index=False)

# TODO add real data on test set

# GAPs
midi_files = list((GAPS_PATH / 'midi').glob("*.mid"))
midi_stems = [midi.stem.replace('-fine-aligned', '') for midi in midi_files]

all_sets = []
audio_files = list((GAPS_PATH / 'audio_16k').glob("*.mp3"))
for audio in audio_files:
    if audio.stem in midi_stems:
        all_sets.append((audio.relative_to(GAPS_PATH), f'midi/{audio.stem}-fine-aligned.mid'))

# select 10% of the all_sets
test_set = random.sample(all_sets, int(len(all_sets) * TESTSET_RATIO))
train_set = [set for set in all_sets if set not in test_set]

# save as csv
train_set_df = pd.DataFrame(train_set, columns=["audio", "label"])
test_set_df = pd.DataFrame(test_set, columns=["audio", "label"])
train_set_df.to_csv(DATASET_PATH / "GAPS_train.csv", index=False)
test_set_df.to_csv(DATASET_PATH / "GAPS_test.csv", index=False)

'''
# Guitarset
all_labels = list((GUITARSET_PATH / 'annotation').glob('*.mid'))
labels = [el for el in all_labels if 'pshift' not in el.stem]

train_labels = random.sample(labels, int(len(labels) * (1-TESTSET_RATIO)))
test_labels = [el for el in labels if el not in train_labels]

wavs = list(GUITARSET_PATH.glob('**/*.wav'))
train_set = []
test_set = []
folder_postfix = [('audio_hex-pickup_debleeded', '_hex_cln'),
                  ('audio_hex-pickup_original', '_hex'),
                  ('audio_mono-mic', '_mic'),
                  ('audio_mono-pickup_mix', '_mix')]

for label in train_labels:
    for folder, postfix in folder_postfix:
        train_set.append((
                          f'{folder}/{label.stem}{postfix}.wav',
                          f'annotation/{label.stem}.mid'))
for label in test_labels:
    for folder, postfix in folder_postfix:
        test_set.append((
                          f'{folder}/{label.stem}{postfix}.wav',
                          f'annotation/{label.stem}.mid'))

train_set_df = pd.DataFrame(train_set, columns=["audio", "label"])
test_set_df = pd.DataFrame(test_set, columns=["audio", "label"])
train_set_df.to_csv(DATASET_PATH / "Guitarset_train.csv", index=False)
test_set_df.to_csv(DATASET_PATH / "Guitarset_test.csv", index=False)






