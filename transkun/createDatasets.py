import pickle
import argparse
import pickle
import os
from pathlib import Path
import csv
import pretty_midi
import soundfile as sf
from . import Data



if __name__ == "__main__":
    
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("outputPath", help = "path to the output folder")
    argumentParser.add_argument("--noPedalExtension", action='store_true', help = "Do not perform pedal extension according to the sustain pedal")


    args = argumentParser.parse_args()

    extendPedal = not args.noPedalExtension
    outputPath = args.outputPath

    maestro_dataset = Data.createDatasetMAESTRO_2('data/maestro-v3.0.0', 'data/maestro-v3.0.0/maestro-v3.0.0.csv', extendSustainPedal = extendPedal)

    train = []
    val= []
    test = []
    for e in maestro_dataset:
        if e["split"] == "train":
            train.append(e)
        elif e["split"] == "validation":
            val.append(e)
        elif e["split"] == "test":
            test.append(e)


    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    with open(os.path.join(outputPath, 'train_maestro.pickle'), 'wb') as f:
        pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, 'val_maestro.pickle'), 'wb') as f:
        pickle.dump(val, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, 'test_maestro.pickle'), 'wb') as f:
        pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)

    '''
    EGDB_train = Data.createDataset_2('data/EGDB_flac', 'data/EGDB_train.csv', extendSustainPedal=extendPedal)
    EGDB_test = Data.createDataset_2('data/EGDB_flac', 'data/EGDB_test.csv', extendSustainPedal=extendPedal)
    with open(os.path.join(outputPath, 'train_EGDB.pickle'), 'wb') as f:
        pickle.dump(EGDB_train, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, 'test_EGDB.pickle'), 'wb') as f:
        pickle.dump(EGDB_test, f, pickle.HIGHEST_PROTOCOL)

    GAPS_train = Data.createDataset_2('data/gaps_v1', 'data/GAPS_train.csv', extendSustainPedal=extendPedal)
    GAPS_test = Data.createDataset_2('data/gaps_v1', 'data/GAPS_test.csv', extendSustainPedal=extendPedal)
    with open(os.path.join(outputPath, 'train_GAPS.pickle'), 'wb') as f:
        pickle.dump(GAPS_train, f, pickle.HIGHEST_PROTOCOL) 
    with open(os.path.join(outputPath, 'test_GAPS.pickle'), 'wb') as f:
        pickle.dump(GAPS_test, f, pickle.HIGHEST_PROTOCOL)

    Guitarset_train = Data.createDataset_2('data/guitarset_yourmt3_16k', 'data/Guitarset_train.csv', extendSustainPedal=extendPedal)
    Guitarset_test = Data.createDataset_2('data/guitarset_yourmt3_16k', 'data/Guitarset_test.csv', extendSustainPedal=extendPedal)
    with open(os.path.join(outputPath, 'train_Guitarset.pickle'), 'wb') as f:
        pickle.dump(Guitarset_train, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(outputPath, 'test_Guitarset.pickle'), 'wb') as f:
        pickle.dump(Guitarset_test, f, pickle.HIGHEST_PROTOCOL)
    '''