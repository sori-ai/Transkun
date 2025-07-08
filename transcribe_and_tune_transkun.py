from collections import defaultdict
import math
import tempfile
from pathlib import Path
import subprocess
import argparse
import bisect
import json
import os

import torch as th
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile

from tqdm import tqdm

# TransKun imports
from transkun.ModelTransformer import TransKun, ModelConfig
from transkun.Data import Note, writeMidi
from transkun.Util import makeFrame

def load_audio(audiofile, target_sr=16000, stop=None):
    """Load audio file and resample to target sample rate"""
    try:
        # Try with soundfile first
        audio, sr = soundfile.read(audiofile, stop=stop)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        # Ensure we have a 1D numpy array
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 0:
            audio = np.array([audio])  # Convert scalar to array
        elif audio.ndim > 1:
            audio = audio.flatten()  # Flatten if somehow still multi-dimensional
        
        # Normalize to [-1, 1] range like pydub does
        if audio.dtype == np.int16:
            audio = np.float32(audio) / 2**15
        elif audio.dtype == np.int32:
            audio = np.float32(audio) / 2**31
        else:
            audio = audio.astype(np.float32)
            
        if sr != target_sr:
            audio = librosa.resample(np.array(audio), orig_sr=sr, target_sr=target_sr)
        
        return np.asarray(audio, dtype=np.float32)
        
    except Exception as e:
        print(f"Error loading {audiofile} with soundfile: {e}, trying with ffmpeg...")
        path_audio = Path(audiofile)
        filetype = path_audio.suffix
        assert filetype in ['.mp3', '.ogg', '.flac', '.wav', '.m4a', '.mp4', '.mov'], filetype
        
        with tempfile.TemporaryDirectory() as tempdir:
            temp_flac = Path(tempdir) / (path_audio.stem + '_temp' + '.flac')
            command = ['ffmpeg', '-i', audiofile, '-ar', str(target_sr), '-ac', '1', '-y', str(temp_flac)] 
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode()}")
                raise RuntimeError(f"FFmpeg failed to convert {audiofile}")
            audio, sr = soundfile.read(temp_flac, stop=stop)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Ensure proper normalization for ffmpeg converted audio
            audio = np.asarray(audio, dtype=np.float32)
        
        return audio

def transcribe_with_transkun(model, audio, device='cuda'):
    """Transcribe audio using TransKun model"""
    print('Transcription Start')
    
    # Convert audio to tensor format expected by TransKun
    # TransKun expects [samples, channels] format (will be transposed internally)
    if audio.ndim == 1:
        # For mono audio, reshape to [samples, 1] to match original readAudio format
        audio_tensor = th.from_numpy(audio).float().unsqueeze(1).to(device)  # [samples, 1]
    else:
        # For multi-channel audio, keep as [samples, channels]
        audio_tensor = th.from_numpy(audio).float().to(device)  # [samples, channels]
    
    with th.no_grad():
        # Use TransKun's transcribe method with model's default parameters
        notes = model.transcribe(audio_tensor, stepInSecond=None, segmentSizeInSecond=None, discardSecondHalf=False)
        print(f'Transcription completed, got {len(notes)} notes')
    
    return notes

def load_transkun_model(model_path, config_path=None, device='cuda'):
    """Load TransKun model from checkpoint"""
    print(f'Loading TransKun model from: {model_path}')
    
    # Auto-detect config path if not provided
    if config_path is None:
        model_dir = Path(model_path).parent
        config_path = model_dir / "2.0.conf"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}. Please provide config_path parameter.")
    
    # Load config using moduleconf like the original TransKun code
    try:
        import moduleconf
        confManager = moduleconf.parseFromFile(str(config_path))
        config = confManager["Model"].config
        
        # Ensure config has the required attributes
        if not hasattr(config, 'fs'):
            raise ValueError("Config missing required 'fs' attribute")
            
    except Exception as e:
        print(f"Failed to load config from {config_path}: {e}")
        print("Falling back to default config...")
        # Create a basic config with required attributes
        config = type('Config', (), {
            'fs': 16000,
            'hopSize': 1024,
            'windowSize': 4096,
            'segmentSizeInSecond': 16,
            'segmentHopSizeInSecond': 8
        })()
    
    # Load checkpoint
    checkpoint = th.load(model_path, map_location='cpu')
    
    # Create model using imported TransKun class
    try:
        model = TransKun(config).to(device)
    except Exception as e:
        print(f"Failed to create model with loaded config: {e}")
        print("Creating model with default config...")
        # Fallback to a simple config
        config = ModelConfig()  # Use the default ModelConfig
        model = TransKun(config).to(device)
    
    # Load state dict
    if 'best_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['best_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model, config

def notes_to_tuples(notes):
    """Convert TransKun Note objects to tuples for detune estimation"""
    note_tuples = []
    for note in notes:
        # Convert to (pitch, (start, end), velocity) format
        note_tuples.append((note.pitch, (note.start, note.end), note.velocity))
    return note_tuples

def estimate_detune(y, sr, notes):
    """Estimate detune from audio and transcribed notes"""
    detunes = []
    
    print(f"DEBUG: Analyzing {len(notes)} notes for detune estimation")
    
    for i, note in enumerate(notes):
        pitch = note.pitch
        start = note.start
        end = note.end
        vel = note.velocity
        
        # More lenient criteria for note selection - especially for detecting larger detunes
        if end - start > 0.15 and vel > 30:  # Even more lenient: 0.15s duration, 30 velocity
            if end - start > 3.0:  # Allow longer segments
                end = start + 3.0
            
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            y_seg = y[start_sample:end_sample]
            len_y = len(y_seg)
            
            if len_y < 128:  # Further reduced minimum length
                print(f"DEBUG: Note {i+1} too short ({len_y} samples), skipping")
                continue
            
            # Use a power-of-2 FFT size for better frequency resolution
            n_fft = 2 ** int(np.ceil(np.log2(len_y)))
            n_fft = min(n_fft, 8192)  # Cap at reasonable size
            
            try:
                S = librosa.stft(y=y_seg, n_fft=n_fft, hop_length=n_fft//4, center=True)
                # Take magnitude spectrum of the first frame (most stable)
                magnitude = np.abs(S[:, 0]) if S.shape[1] > 0 else np.abs(S).mean(axis=1)
                
                # Define frequency range around the expected pitch - more generous for larger detunes
                expected_freq = librosa.midi_to_hz(pitch)
                f_min = expected_freq * 0.8  # Allow 20% deviation (more generous)
                f_max = expected_freq * 1.2
                
                freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
                freq_left = bisect.bisect_left(freqs, f_min)
                freq_right = bisect.bisect_right(freqs, f_max)
                
                if freq_right <= freq_left + 1:  # Need at least 2 frequency bins
                    print(f"DEBUG: Note {i+1} frequency range too narrow, skipping")
                    continue
                
                # Extract magnitude in the frequency range of interest
                x = magnitude[freq_left:freq_right+1]
                
                if len(x) == 0 or np.max(x) == 0:
                    print(f"DEBUG: Note {i+1} no energy in frequency range, skipping")
                    continue
                
                # Find peak with sub-bin precision using parabolic interpolation
                peak_idx = np.argmax(x)
                
                # Parabolic interpolation for sub-bin accuracy
                if peak_idx > 0 and peak_idx < len(x) - 1:
                    # Use parabolic interpolation
                    y1, y2, y3 = x[peak_idx-1], x[peak_idx], x[peak_idx+1]
                    a = (y1 - 2*y2 + y3) / 2
                    b = (y3 - y1) / 2
                    if a != 0:
                        x_offset = -b / (2*a)
                        peak_freq = freqs[freq_left + peak_idx] + x_offset * (freqs[1] - freqs[0])
                    else:
                        peak_freq = freqs[freq_left + peak_idx]
                else:
                    # Use simple weighted average for edge cases
                    if peak_idx == 0 and len(x) > 1:
                        weights = x[0:2]
                        peak_freq = np.average(freqs[freq_left:freq_left+2], weights=weights)
                    elif peak_idx == len(x) - 1 and len(x) > 1:
                        weights = x[-2:]
                        peak_freq = np.average(freqs[freq_left+len(x)-2:freq_left+len(x)], weights=weights)
                    else:
                        peak_freq = freqs[freq_left + peak_idx]
                
                # Calculate detune in cents
                detune = 1200 * np.log2(peak_freq / expected_freq)
                
                # Accept a wider range of detunes to detect larger tuning issues
                if abs(detune) < 150:  # Increased from 100 to 150 cents (1.5 semitones)
                    detunes.append(detune)
                    print(f"DEBUG: Note {i+1} (pitch={pitch}, dur={end-start:.2f}s, vel={vel}): "
                          f"expected={expected_freq:.1f}Hz, found={peak_freq:.1f}Hz, "
                          f"detune={detune:.1f} cents")
                else:
                    print(f"DEBUG: Note {i+1} detune too large ({detune:.1f} cents), skipping")
                    
            except Exception as e:
                print(f"DEBUG: Error processing note {i+1}: {e}")
                continue
    
    if detunes:
        median_detune = np.median(detunes)
        mean_detune = np.mean(detunes)
        print(f"DEBUG: Found {len(detunes)} valid notes for detune estimation")
        print(f"DEBUG: Detune values: {[f'{d:.1f}' for d in detunes]}")
        print(f"DEBUG: Median detune: {median_detune:.1f} cents")
        print(f"DEBUG: Mean detune: {mean_detune:.1f} cents")
        return median_detune
    else:
        print("DEBUG: No valid notes found for detune estimation")
        return 0.0

def save_midi_transkun(notes, save_path):
    """Save notes using TransKun's MIDI writing function"""
    midi_obj = writeMidi(notes)
    midi_obj.write(str(save_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to TransKun model checkpoint')
    parser.add_argument('--audio_path', type=str, help='Path to single audio file')
    parser.add_argument('--target_folder', type=str, help='Path to folder containing audio files')
    parser.add_argument('--save_folder', type=Path, help='Output folder for MIDI files')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length in seconds for detune estimation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    assert args.audio_path is not None or args.target_folder is not None, 'Either audio_path or target_folder should be specified'
     
    # Determine audio files to process
    if args.audio_path:
        audio_files = [Path(args.audio_path)]
        result_txt = Path(args.audio_path).parent / 'detune.txt'
        results = []
    else:
        # Find audio files in target folder
        audio_files = []
        for ext in ['*.flac', '*.wav', '*.mp3', '*.m4a', '*.ogg']:
            audio_files.extend(Path(args.target_folder).glob(f'**/{ext}'))
        result_txt = Path(args.target_folder) / 'detune.txt'
        results = []

    # Load TransKun model
    model, config = load_transkun_model(args.model_path, device=args.device)
    
    # Ensure config has fs attribute (fallback to 16000 if missing)
    sample_rate = getattr(config, 'fs', 16000)
    print(f'Loaded TransKun model with sample rate: {sample_rate}Hz')

    for audio_path in tqdm(audio_files):
        print(f'\nProcessing: {audio_path.stem}')
        
        # Load audio for detune estimation (shorter version)
        max_samples = args.max_len * sample_rate
        audio_short = load_audio(audio_path, target_sr=sample_rate, stop=max_samples)
        print(f'DEBUG: Short audio length: {len(audio_short)/sample_rate:.2f} seconds')
        
        # Transcribe for detune estimation
        notes_short = transcribe_with_transkun(model, audio_short, args.device)
        
        # Estimate detune
        detune = estimate_detune(audio_short, sample_rate, notes_short)
        results.append((audio_path, detune))
        print(f'{audio_path.stem} detune: {detune:2.1f} cents')
        
        # Load full audio for final transcription
        audio_full = load_audio(audio_path, target_sr=sample_rate)
        print(f'DEBUG: Full audio length: {len(audio_full)/sample_rate:.2f} seconds')
        
        # Apply detune compensation if needed
        compensation_factor = 1.0  # Track compensation for time adjustment
        if abs(detune) > 5:  # Apply compensation for detune > 5 cents
            print(f'DEBUG: Applying detune compensation of {detune:.1f} cents')
            compensation_speed = 2**(detune/1200)
            compensation_factor = compensation_speed
            audio_full = librosa.resample(audio_full, orig_sr=sample_rate, target_sr=float(sample_rate*compensation_speed))
            print(f'DEBUG: Audio after compensation: {len(audio_full)/sample_rate:.2f} seconds')
        else:
            print(f'DEBUG: Detune too small ({detune:.1f} cents), no compensation applied')
        
        # Final transcription
        notes_full = transcribe_with_transkun(model, audio_full, args.device)
        
        # Restore original note timings if detune compensation was applied
        if compensation_factor != 1.0:
            print(f'DEBUG: Restoring note timings (scaling by {1/compensation_factor:.6f})')
            for note in notes_full:
                note.start = note.start / compensation_factor
                note.end = note.end / compensation_factor
            print(f'DEBUG: Note timings restored to original durations')
        
        # Save MIDI
        if args.save_folder:
            save_path = args.save_folder / f'{audio_path.stem}.mid'
            args.save_folder.mkdir(exist_ok=True)
        else:
            save_path = audio_path.parent / f'{audio_path.stem}.mid'
        
        save_midi_transkun(notes_full, save_path)
        print(f'Saved MIDI: {save_path}')
        
    # Save detune results
    if args.target_folder:
        with open(result_txt, 'w') as f:
            for result in results:
                f.write(f'{result[0]} detune: {result[1]:2.1f} cents\n')
        print(f'Detune results saved to: {result_txt}') 