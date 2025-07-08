import argparse
import torch
import numpy as np
import pickle
import sys
from pathlib import Path
import json
import scipy.signal
from torch.utils.data import DataLoader
import soundfile as sf
import pretty_midi

# Add transkun to path
sys.path.insert(0, str(Path.cwd()))

from transkun.ModelTransformer import *
from transkun.Evaluation import *
from transkun.data.event import *
from evaluate_midi import get_evaluation_metrics
from transkun.Util import makeFrame
from transkun.Data import resolveOverlapping, writeMidi

# Configuration
MODEL_CONF_PATH = "checkpoint/conf2.0.json"
CHECKPOINT_PATH = "transkun/pretrained/2.0.pt"
DATASET_PATH = "data_selected/validation_maestro.pickle"
NOISE_TYPE = "white"  # or 'pink'
SNR_DB = 20.0
SEGMENT_SEC = 8.0  # The size of each audio segment to process
HOP_SEC = 4.0      # The step size between segments (creates 50% overlap)

def inject_noise(audio, noise_type, snr_db):
    """Injects white or pink noise into an audio signal."""
    if audio.size == 0:
        return audio
    signal_power = np.mean(audio**2)
    if signal_power == 0:
        return audio
    signal_db = 10 * np.log10(signal_power)
    noise_db = signal_db - snr_db
    noise_power = 10**(noise_db / 10)
    
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    elif noise_type == 'pink':
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        zi = np.random.normal(0, 1, size=(max(len(a), len(b)) - 1,))
        pink_noise, _ = scipy.signal.lfilter(b, a, np.random.normal(0, 1, audio.shape[0]), zi=zi)
        noise_power_pink = np.mean(pink_noise**2)
        if noise_power_pink == 0:
            return audio
        noise = pink_noise * np.sqrt(noise_power / noise_power_pink)
    else:
        raise ValueError("Unsupported noise type")
        
    return audio + noise.astype(np.float32)

def split_audio_into_chunks(audio, chunk_size_in_seconds, fs):
    """Splits audio into chunks of a specified size."""
    chunk_size_in_samples = int(chunk_size_in_seconds * fs)
    if len(audio) == 0:
        return []
    if len(audio) <= chunk_size_in_samples:
        return [audio]
    
    num_chunks = int(np.ceil(len(audio) / chunk_size_in_samples))
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size_in_samples
        end = start + chunk_size_in_samples
        chunks.append(audio[start:end])
    return chunks

def notes_to_array(notes):
    """Converts a list of Note objects to a numpy array (start, end, pitch, velocity)."""
    note_list = []
    if not notes:
        return np.empty((0, 4))
    for note in notes:
        # mir_eval only handles positive pitch values for notes.
        # Negative values are used by the model for special events like pedals.
        if note.pitch > 0:
            note_list.append([note.start, note.end, note.pitch, note.velocity])
    
    if not note_list:
        return np.empty((0, 4))
        
    return np.array(note_list)

class ListDatasetWrapper:
    def __init__(self, data, target_fs):
        self.data = [item for item in data if item.get('duration', 0) > 0 and item.get('audio_filename')]
        self.durations = [item.get('duration') for item in self.data]
        self.target_fs = target_fs
        print(f"Loaded {len(self.data)} samples with duration > 0 and an audio file.")

    def __len__(self):
        return len(self.data)

    def get_sample(self, idx):
        sample = self.data[idx]
        notes = sample.get('notes', [])
        audio_path = sample.get('audio_filename')
        fs = sample.get('fs', 16000)
        
        audio, fs_read = sf.read(audio_path, dtype='float32')
        assert fs == fs_read
        
        if fs != self.target_fs:
            num_samples = int(len(audio) * self.target_fs / fs)
            resampled_audio = scipy.signal.resample(audio, num_samples)
            if isinstance(resampled_audio, tuple):
                audio = np.array(resampled_audio[0], dtype=np.float32)
            else:
                audio = np.array(resampled_audio, dtype=np.float32)
            fs = self.target_fs
            
        return notes, audio, fs

def evaluate_file(notes_pred, notes_gt):
    """
    Evaluates transcription accuracy against ground truth using the shared evaluation function.
    """
    ref_array = notes_to_array(notes_gt)
    est_array = notes_to_array(notes_pred)
    
    return get_evaluation_metrics(ref_array, est_array)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(MODEL_CONF_PATH) as f:
        model_conf = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

    model = TransKun(conf=model_conf.Model.config).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    state_dict_key = 'model' if 'model' in checkpoint else 'state_dict'
    if state_dict_key in checkpoint:
        model.load_state_dict(checkpoint[state_dict_key])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully.")

    with open(DATASET_PATH, 'rb') as f:
        dataset_list = pickle.load(f)
    dataset = ListDatasetWrapper(dataset_list, model.fs)

    total_metrics = {
        'Onset': {'Precision': [], 'Recall': [], 'F1-score': []},
        'Offset': {'Precision': [], 'Recall': [], 'F1-score': []},
        'Velocity': {'Precision': [], 'Recall': [], 'F1-score': []}
    }

    for i in range(len(dataset)):
        notes_gt, audio, fs = dataset.get_sample(i)
        
        audio_noisy = inject_noise(audio, NOISE_TYPE, SNR_DB)
        
        # Save the first noisy sample
        if i == 0:
            sf.write("noisy_sample_white.wav", audio_noisy, fs)
            print("Saved noisy_sample.wav")

        notes_pred = []
        
        audio_len_samples = len(audio_noisy)
        segment_len_samples = int(SEGMENT_SEC * fs)
        hop_len_samples = int(HOP_SEC * fs)
        
        # This holds the state for the CRF decoder between chunks
        startPos = None

        for start_sample in range(0, audio_len_samples, hop_len_samples):
            end_sample = start_sample + segment_len_samples
            chunk = audio_noisy[start_sample:end_sample]

            # Skip chunk if it's too short (often the last one)
            if len(chunk) < model.windowSize:
                continue

            current_time_offset = start_sample / fs

            audio_chunk_tensor = torch.from_numpy(chunk).float()
            if audio_chunk_tensor.dim() == 1:
                # Add a channel dimension for mono audio -> [1, samples]
                audio_chunk_tensor = audio_chunk_tensor.unsqueeze(0)
            
            # Convert audio chunk to frames -> [channel, frames, window_size]
            frames = makeFrame(audio_chunk_tensor, model.hopSize, model.windowSize)
            
            # Add a batch dimension -> [1, channel, frames, window_size]
            frames = frames.unsqueeze(0)
            frames_tensor = frames.to(device)

            with torch.no_grad():
                notes_pred_chunk, lastP = model.transcribeFrames(frames_tensor, forcedStartPos=startPos)
                # transcribeFrames returns a list of note lists, one for each item in the batch
                notes_pred_chunk = notes_pred_chunk[0]
            
            # Update the start position for the next chunk's decoder
            hop_len_frames = hop_len_samples / model.hopSize
            startPos = [max(int(p - hop_len_frames), 0) for p in lastP]
            
            for note in notes_pred_chunk:
                note.start += current_time_offset
                note.end += current_time_offset
                notes_pred.append(note)

            del audio_chunk_tensor, frames, frames_tensor, notes_pred_chunk

        # Resolve notes predicted in overlapping segments
        notes_pred = resolveOverlapping(notes_pred)

        # Save the first transcribed midi
        if i == 0:
            midi_pred = writeMidi(notes_pred)
            midi_pred.write("transcribed_sample_white.mid")
            print("Saved transcribed_sample.mid")

        metrics = evaluate_file(notes_pred, notes_gt)
        
        for metric_type, values in metrics.items():
            key_name = metric_type.capitalize()
            total_metrics[key_name]['Precision'].append(values[0])
            total_metrics[key_name]['Recall'].append(values[1])
            total_metrics[key_name]['F1-score'].append(values[2])

        print(f"Processed sample {i+1}/{len(dataset)}: Onset F1={metrics['onset'][2]:.4f}, Offset F1={metrics['offset'][2]:.4f}, Velocity F1={metrics['velocity'][2]:.4f}")

        # Unload data to free up memory for the next iteration
        del notes_gt, audio, audio_noisy, notes_pred, metrics

    print("\n" + "="*30)
    print(f"Average Validation Metrics with {NOISE_TYPE} noise @ {SNR_DB} dB SNR")
    print("="*30)
    for metric_type, type_metrics in total_metrics.items():
        print(f"  {metric_type}:")
        for key, values in type_metrics.items():
            avg_metric = np.mean(values)
            print(f"    {key}: {avg_metric:.4f}")
    print("="*30 + "\n")

if __name__ == '__main__':
    main() 