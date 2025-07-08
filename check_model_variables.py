#!/usr/bin/env python3
"""
TransKun Model Variables Checker - Inner Product Score Analysis

This script helps you check and analyze the inner product score with input audio file.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import json
from collections import OrderedDict
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pretty_midi
import random
from utils.midi import midi2note

# Import TransKun model
try:
    from transkun.ModelTransformer import TransKun, ModelConfig
    print("✓ Successfully imported TransKun model")
except ImportError as e:
    print(f"✗ Error importing TransKun model: {e}")
    sys.exit(1)

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    print(f"\n=== SETTING RANDOM SEED: {seed} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("✓ Random seed set for reproducibility")

def load_configuration():
    """Load model configuration from checkpoint"""
    print("\n=== LOADING CONFIGURATION ===")
    
    config_paths = [
        'checkpoint/conf2.0.json',  # New config that matches pretrained model
        'checkpoint/conf.json',
        'run/2.0.conf',
        'transkun/pretrained/2.0.conf'
    ]
    
    config_data = None
    used_path = None
    
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config_data = json.load(f)
                used_path = path
                print(f"✓ Configuration loaded from: {path}")
                break
            except Exception as e:
                print(f"✗ Error loading {path}: {e}")
    
    if config_data is None:
        print("⚠ No configuration file found, using default config")
        return None, None
    
    return config_data, used_path

def create_model_config(config_data=None):
    """Create model configuration"""
    print("\n=== CREATING MODEL CONFIG ===")
    
    config = ModelConfig()
    
    if config_data and 'Model' in config_data and 'config' in config_data['Model']:
        checkpoint_config = config_data['Model']['config']
        print("Updating config with checkpoint values:")
        for key, value in checkpoint_config.items():
            if hasattr(config, key):
                old_value = getattr(config, key)
                setattr(config, key, value)
                print(f"  {key}: {old_value} → {value}")
    
    print("\nFinal Model Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    return config

def load_audio_file(audio_path):
    """Load and preprocess audio file"""
    print(f"\n=== LOADING AUDIO FILE: {audio_path} ===")
    
    if not os.path.exists(audio_path):
        print(f"✗ Audio file not found: {audio_path}")
        return None
    
    try:
        # Only load the first 30 seconds
        audio, sr = librosa.load(audio_path, sr=None, duration=30.0)
        print("✓ Audio loaded successfully")
        print(f"  Duration: {len(audio)/sr:.2f} seconds (limited to 30s)")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Shape: {audio.shape}")
        
        # Resample if necessary
        target_sr = 44100  # Default sample rate
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            print(f"  Resampled to {target_sr} Hz")
        
        return audio, target_sr
        
    except Exception as e:
        print(f"✗ Error loading audio: {e}")
        return None

def analyze_inner_product_score(model, audio, sr):
    """Analyze inner product score with the audio input"""
    print("\n=== ANALYZING INNER PRODUCT SCORE ===")
    
    try:
        # Prepare audio input
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, n_samples]
        print(f"Audio tensor shape: {audio_tensor.shape}")
        
        # Create frames
        hop_size = model.hopSize
        window_size = model.windowSize
        
        # Calculate number of frames
        n_samples = audio_tensor.shape[-1]
        n_frames = (n_samples - window_size) // hop_size + 1
        
        print(f"Number of frames: {n_frames}")
        print(f"Frame duration: {window_size/sr:.3f} seconds")
        print(f"Hop duration: {hop_size/sr:.3f} seconds")
        
        # Create frames manually
        frames = []
        for i in range(n_frames):
            start_idx = i * hop_size
            end_idx = start_idx + window_size
            frame = audio_tensor[:, :, start_idx:end_idx]
            frames.append(frame)
        
        frames_batch = torch.stack(frames, dim=2)  # [1, 1, n_frames, window_size]
        print(f"Frames batch shape: {frames_batch.shape}")
        
        # Process frames through the model
        with torch.no_grad():
            # Feature extraction
            features = model.framewiseFeatureExtractor(frames_batch)
            print(f"Features shape: {features.shape}")
            # Fix: squeeze channel dimension if needed
            if features.shape[1] == 1:
                features = features.squeeze(1)
            print(f"Features shape after squeeze: {features.shape}")
            
            # Backbone processing
            target_pitches = torch.tensor(model.targetMIDIPitch, device=features.device)
            ctx = model.backbone(features, outputIndices=target_pitches)
            print(f"Context shape: {ctx.shape}")
            
            # Inner product scoring
            if model.useInnerProductScorer:
                S_batch, S_skip_batch = model.scorer(ctx)
                print("S_batch stats: min", S_batch.min().item(), "max", S_batch.max().item(), "mean", S_batch.mean().item(), "std", S_batch.std().item())
                print("Any NaNs in S_batch?", torch.isnan(S_batch).any().item())
                print("Any Infs in S_batch?", torch.isinf(S_batch).any().item())
                print("S_skip_batch stats: min", S_skip_batch.min().item(), "max", S_skip_batch.max().item(), "mean", S_skip_batch.mean().item(), "std", S_skip_batch.std().item())
                print("Any NaNs in S_skip_batch?", torch.isnan(S_skip_batch).any().item())
                print("Any Infs in S_skip_batch?", torch.isinf(S_skip_batch).any().item())
                
                # Analyze score statistics
                print(f"\nScore Statistics:")
                print(f"  Score batch - Min: {S_batch.min().item():.4f}, Max: {S_batch.max().item():.4f}, Mean: {S_batch.mean().item():.4f}")
                print(f"  Skip score batch - Min: {S_skip_batch.min().item():.4f}, Max: {S_skip_batch.max().item():.4f}, Mean: {S_skip_batch.mean().item():.4f}")
                
                # Analyze score distribution for different pitch classes
                print(f"\nScore Analysis by Pitch Class:")
                n_pitches = S_batch.shape[-1]
                for i in range(n_pitches):
                    pitch_scores = S_batch[:, :, 0, i].flatten()
                    print(f"  Pitch {model.targetMIDIPitch[i]}: Min={pitch_scores.min():.4f}, Max={pitch_scores.max():.4f}, Mean={pitch_scores.mean():.4f}")
                
                # --- Only generate score_matrix_20pitches_overlay_1200_midi.png with MIDI overlay and top 20 stars ---
                midi_path = 'gymnopedie1_transcribed.mid'  # Change if needed
                midi_data = pretty_midi.PrettyMIDI(midi_path)
                window_size = 400
                valid_pitch_tuples = [(i, p) for i, p in enumerate(model.targetMIDIPitch) if 21 <= p <= 108]
                if len(valid_pitch_tuples) == 0:
                    print('No valid MIDI pitches (21-108) found in model.targetMIDIPitch!')
                else:
                    n_overlay_pitches = min(20, len(valid_pitch_tuples))
                    selected_indices = np.linspace(0, len(valid_pitch_tuples)-1, n_overlay_pitches, dtype=int)
                    pitch_indices_overlay = [valid_pitch_tuples[i][0] for i in selected_indices]
                    midi_pitches_overlay = [valid_pitch_tuples[i][1] for i in selected_indices]
                    midi_to_name = lambda midi: f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][(midi%12)]}{(midi//12)-1}"
                    # Build a dict: midi_pitch -> list of (onset_frame, offset_frame)
                    midi_notes_by_pitch = {p: [] for p in midi_pitches_overlay}
                    hop_size = model.hopSize
                    fs = model.fs
                    for inst in midi_data.instruments:
                        for note in inst.notes:
                            if note.pitch in midi_notes_by_pitch:
                                onset_frame = int(note.start * fs // hop_size)
                                offset_frame = int(note.end * fs // hop_size)
                                midi_notes_by_pitch[note.pitch].append((onset_frame, offset_frame))
                    n_cols = 5
                    n_rows = 4
                    # Use fixed colorbar range for all 20-pitch plots
                    fixed_vmin = -15
                    fixed_vmax = 10
                    # Linear scale plot
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
                    axes = axes.flatten()
                    for idx, (pitch_idx, midi) in enumerate(zip(pitch_indices_overlay, midi_pitches_overlay)):
                        pitch_name = midi_to_name(midi)
                        score_matrix_win = S_batch[:window_size, :window_size, 0, pitch_idx].cpu().numpy()
                        mask = np.tri(*score_matrix_win.shape, k=0, dtype=bool)
                        score_matrix_win[mask] = np.nan
                        score_flat = score_matrix_win.flatten()
                        valid_indices = np.where(~np.isnan(score_flat))[0]
                        n_values = min(20, len(valid_indices))
                        top_indices = valid_indices[np.argsort(score_flat[valid_indices])[-n_values:]] if n_values > 0 else []
                        top_coords = np.column_stack(np.unravel_index(top_indices, score_matrix_win.shape)) if n_values > 0 else np.empty((0,2), dtype=int)
                        midi_intervals = [(i, j) for (i, j) in midi_notes_by_pitch[midi] if i < window_size and j < window_size]
                        ax = axes[idx]
                        im = ax.imshow(score_matrix_win, aspect='auto', origin='upper', cmap='coolwarm', vmin=fixed_vmin, vmax=fixed_vmax)
                        ax.set_title(f'{pitch_name} (MIDI {midi})')
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Frame')
                        if top_coords.shape[0] > 0:
                            ax.scatter(top_coords[:,1], top_coords[:,0], s=80, marker='*', color='purple', alpha=0.7, label='Top 20 values')
                        for k, (i, j) in enumerate(midi_intervals):
                            ax.scatter(j, i, s=100, marker='o', color='red', edgecolors='white', linewidth=1, label='MIDI note' if k == 0 else None)
                        if idx == 0:
                            handles, labels = ax.get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
                    for j in range(idx+1, len(axes)):
                        axes[j].axis('off')
                    fig.tight_layout(rect=(0, 0, 0.95, 1))
                    cbar_ax = fig.add_axes((0.96, 0.15, 0.02, 0.7))
                    fig.colorbar(im, cax=cbar_ax, label='Score Value')
                    plt.savefig('score_matrix_20pitches_overlay_1200_midi.png', dpi=300, bbox_inches='tight')
                    # Log-scale version
                    fig_log, axes_log = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
                    axes_log = axes_log.flatten()
                    for idx, (pitch_idx, midi) in enumerate(zip(pitch_indices_overlay, midi_pitches_overlay)):
                        pitch_name = midi_to_name(midi)
                        score_matrix_win = S_batch[:window_size, :window_size, 0, pitch_idx].cpu().numpy()
                        mask = np.tri(*score_matrix_win.shape, k=0, dtype=bool)
                        score_matrix_win[mask] = np.nan
                        score_matrix_log = np.sign(score_matrix_win) * np.log1p(np.abs(score_matrix_win))
                        score_flat = score_matrix_log.flatten()
                        valid_indices = np.where(~np.isnan(score_flat))[0]
                        n_values = min(20, len(valid_indices))
                        top_indices = valid_indices[np.argsort(score_flat[valid_indices])[-n_values:]] if n_values > 0 else []
                        top_coords = np.column_stack(np.unravel_index(top_indices, score_matrix_log.shape)) if n_values > 0 else np.empty((0,2), dtype=int)
                        midi_intervals = [(i, j) for (i, j) in midi_notes_by_pitch[midi] if i < window_size and j < window_size]
                        ax = axes_log[idx]
                        im_log = ax.imshow(score_matrix_log, aspect='auto', origin='upper', cmap='coolwarm', vmin=fixed_vmin, vmax=fixed_vmax)
                        ax.set_title(f'{pitch_name} (MIDI {midi}) [Log]')
                        ax.set_xlabel('Frame')
                        ax.set_ylabel('Frame')
                        if top_coords.shape[0] > 0:
                            ax.scatter(top_coords[:,1], top_coords[:,0], s=80, marker='*', color='purple', alpha=0.7, label='Top 20 values')
                        for k, (i, j) in enumerate(midi_intervals):
                            ax.scatter(j, i, s=100, marker='o', color='red', edgecolors='white', linewidth=1, label='MIDI note' if k == 0 else None)
                        if idx == 0:
                            handles, labels = ax.get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
                    for j in range(idx+1, len(axes_log)):
                        axes_log[j].axis('off')
                    fig_log.tight_layout(rect=(0, 0, 0.95, 1))
                    cbar_ax_log = fig_log.add_axes((0.96, 0.15, 0.02, 0.7))
                    fig_log.colorbar(im_log, cax=cbar_ax_log, label='Signed Log Score Value')
                    plt.savefig('score_matrix_20pitches_overlay_1200_midi_log.png', dpi=300, bbox_inches='tight')
                    plt.show()
                
                # Print score values at true note locations for pitch 66
                if S_batch is not None:
                    pitch = 66
                    try:
                        pitch_index = model.targetMIDIPitch.index(pitch)
                    except ValueError:
                        print(f"Pitch {pitch} not found in model.targetMIDIPitch!")
                        pitch_index = None
                    print(f"[DEBUG] Plotting score matrix for pitch {pitch} (index {pitch_index})")
                    print(f"[DEBUG] S_batch shape: {S_batch.shape}")
                    if pitch_index is not None:
                        score_matrix = S_batch[:, :, 0, pitch_index].cpu().numpy()
                        note_scores = []
                        for inst in midi_data.instruments:
                            for note in inst.notes:
                                if note.pitch == pitch:
                                    onset_frame = int(note.start * fs // hop_size)
                                    offset_frame = int(note.end * fs // hop_size)
                                    if (0 <= onset_frame < score_matrix.shape[0] and
                                        0 <= offset_frame < score_matrix.shape[1]):
                                        score = score_matrix[onset_frame, offset_frame]
                                        note_scores.append(score)
                        print(f"\nScore values at true note locations for pitch {pitch}:")
                        print(note_scores)
                        if note_scores:
                            print(f"Mean score at true notes: {np.mean(note_scores):.4f}")
                            print(f"Std score at true notes: {np.std(note_scores):.4f}")
                            print(f"Max score at true notes: {np.max(note_scores):.4f}")
                            print(f"Min score at true notes: {np.min(note_scores):.4f}")
                        print(f"Global score matrix stats: min={np.nanmin(score_matrix):.4f}, max={np.nanmax(score_matrix):.4f}, mean={np.nanmean(score_matrix):.4f}, std={np.nanstd(score_matrix):.4f}")
                
                return S_batch, S_skip_batch, ctx
            else:
                print("✗ Model is not using inner product scorer")
                return None, None, ctx
                
    except Exception as e:
        print(f"✗ Error analyzing inner product score: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def visualize_scores(S_batch, S_skip_batch, model, save_path="score_visualization.png"):
    """Visualize the inner product scores"""
    print(f"\n=== VISUALIZING SCORES ===")
    
    try:
        if S_batch is None:
            print("✗ No scores to visualize")
            return
        
        # Convert to numpy for visualization
        S_batch_np = S_batch.cpu().numpy()
        S_skip_batch_np = S_skip_batch.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Score heatmap for a specific pitch
        pitch_idx = 60  # Middle C
        if pitch_idx < len(model.targetMIDIPitch):
            pitch = model.targetMIDIPitch[pitch_idx]
            score_pitch = S_batch_np[:, :, pitch_idx]
            im1 = axes[0, 0].imshow(score_pitch.T, aspect='auto', cmap='viridis')
            axes[0, 0].set_title(f'Score Heatmap for Pitch {pitch}')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Time Step')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Score distribution
        axes[0, 1].hist(S_batch_np.flatten(), bins=50, alpha=0.7, label='Score Batch')
        axes[0, 1].hist(S_skip_batch_np.flatten(), bins=50, alpha=0.7, label='Skip Score Batch')
        axes[0, 1].set_title('Score Distribution')
        axes[0, 1].set_xlabel('Score Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Mean score over time for different pitch ranges
        n_pitches = len(model.targetMIDIPitch)
        pitch_ranges = [
            (0, n_pitches//4, 'Low Pitches'),
            (n_pitches//4, n_pitches//2, 'Mid-Low Pitches'),
            (n_pitches//2, 3*n_pitches//4, 'Mid-High Pitches'),
            (3*n_pitches//4, n_pitches, 'High Pitches')
        ]
        
        for i, (start, end, label) in enumerate(pitch_ranges):
            mean_scores = S_batch_np[:, :, start:end].mean(axis=(1, 2))
            axes[1, 0].plot(mean_scores, label=label, alpha=0.7)
        
        axes[1, 0].set_title('Mean Scores Over Time by Pitch Range')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Mean Score')
        axes[1, 0].legend()
        
        # 4. Score variance across pitches
        score_variance = S_batch_np.var(axis=(0, 1))
        pitch_indices = range(len(model.targetMIDIPitch))
        axes[1, 1].plot(pitch_indices, score_variance)
        axes[1, 1].set_title('Score Variance Across Pitches')
        axes[1, 1].set_xlabel('Pitch Index')
        axes[1, 1].set_ylabel('Variance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
        
    except Exception as e:
        print(f"✗ Error visualizing scores: {e}")
        import traceback
        traceback.print_exc()

def analyze_model_components(model):
    """Analyze all components and variables in the model"""
    print("\n=== MODEL COMPONENTS ANALYSIS ===")
    
    # Basic model attributes
    print("\n1. Basic Model Attributes:")
    basic_attrs = ['hopSize', 'windowSize', 'fs', 'segmentSizeInSecond', 'segmentHopSizeInSecond']
    for attr in basic_attrs:
        if hasattr(model, attr):
            print(f"   {attr}: {getattr(model, attr)}")
    
    print(f"\n2. Target MIDI Pitch Range: {model.targetMIDIPitch}")
    print(f"   Number of pitch classes: {len(model.targetMIDIPitch)}")
    print(f"   Pitch range: {min(model.targetMIDIPitch)} to {max(model.targetMIDIPitch)}")
    
    # Feature extractor
    print(f"\n3. Feature Extractor:")
    print(f"   Type: {type(model.framewiseFeatureExtractor).__name__}")
    print(f"   Output channels: {model.framewiseFeatureExtractor.nChannel}")
    print(f"   Output dimensions: {model.framewiseFeatureExtractor.outputDim}")
    
    # Scorer
    print(f"\n4. Scorer:")
    print(f"   Type: {type(model.scorer).__name__}")
    print(f"   Use Inner Product Scorer: {model.useInnerProductScorer}")
    
    # Predictors
    print(f"\n5. Velocity Predictor:")
    print(f"   Type: {type(model.velocityPredictor).__name__}")
    print(f"   Output size: 128 (velocity range)")
    
    print(f"\n6. Refined Onset/Offset Predictor:")
    print(f"   Type: {type(model.refinedOFPredictor).__name__}")
    print(f"   Output size: 4 (2 for onset offset + 2 for presence)")
    
    # Backbone
    print(f"\n7. Backbone:")
    print(f"   Type: {type(model.backbone).__name__}")
    
    return model

def load_pretrained_weights(model):
    """Load pretrained weights if available"""
    print("\n=== LOADING PRETRAINED WEIGHTS ===")
    
    checkpoint_paths = [
        'transkun/pretrained/2.0.pt',
        'checkpoint/model.pt',
        'checkpoint/checkpoint.pt'
    ]
    
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            try:
                print(f"Trying to load checkpoint from: {checkpoint_path}")
                if torch.cuda.is_available():
                    checkpoint = torch.load(checkpoint_path, map_location='cuda')
                else:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print(f"\u2713 Checkpoint loaded from {checkpoint_path}")
                print("Model config:", model.conf.__dict__)
                print("Model state dict keys:", list(model.state_dict().keys())[:10])
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                print("Checkpoint keys:", list(state_dict.keys())[:10])
                
                # Load weights
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print("✓ Weights loaded successfully")
                
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
                
                return model
            except Exception as e:
                print(f"✗ Error loading checkpoint from {checkpoint_path}: {e}")
    
    print("⚠ No valid checkpoint found, using random weights")
    return model

def plot_score(S_batch, model, frame_start, frame_end, midi_path, pitch, save_path=None, std_multiplier=1.5, top_n=100, audio=None):
    """Plot the score matrix for a given pitch from frame_start to frame_end, overlaying the top N values, points above mean + std_multiplier*std, and the played notes as red circles. Optionally overlay Viterbi path using audio."""
    try:
        pitch_index = model.targetMIDIPitch.index(pitch)
    except ValueError:
        print(f"Pitch {pitch} not found in model.targetMIDIPitch!")
        return
    
    if save_path is None:
        save_path = f'score_{pitch}_{frame_start}_{frame_end}.png'
    
    score_matrix = S_batch[frame_start:frame_end, frame_start:frame_end, 0, pitch_index].cpu().numpy()
    # Set diagonal and lower triangle to np.nan
    mask = np.tri(*score_matrix.shape, k=0, dtype=bool)  # True for i >= j
    score_matrix[mask] = np.nan
    score_flat = score_matrix.flatten()
    # Only consider non-nan values for top N
    valid_indices = np.where(~np.isnan(score_flat))[0]
    if valid_indices.size > 0:
        # Get top N values (or all if less than N)
        n_values = min(top_n, len(valid_indices))
        top_indices = valid_indices[np.argsort(score_flat[valid_indices])[-n_values:]]
        top_coords = np.column_stack(np.unravel_index(top_indices, score_matrix.shape))
        
        # Calculate threshold for points above mean + std_multiplier * std
        valid_scores = score_flat[valid_indices]
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores)
        threshold = mean_score + std_multiplier * std_score
        
        # Find points above threshold
        above_threshold_indices = valid_indices[valid_scores > threshold]
        above_threshold_coords = np.column_stack(np.unravel_index(above_threshold_indices, score_matrix.shape))
    else:
        top_coords = np.empty((0, 2), dtype=int)
        above_threshold_coords = np.empty((0, 2), dtype=int)
    
    plt.figure(figsize=(8, 8))
    im = plt.imshow(score_matrix, aspect='auto', origin='upper', cmap='coolwarm')
    plt.title(f'Score Matrix for Pitch {pitch}, Frames {frame_start}-{frame_end} (Upper Triangle Only)')
    plt.xlabel('Frame')
    plt.ylabel('Frame')
    
    # Plot top N values
    if top_coords.shape[0] > 0:
        plt.scatter(top_coords[:,1], top_coords[:,0], s=80, marker='*', color='purple', alpha=0.7, label=f'Top {n_values} values')
        print(f"Plotted {top_coords.shape[0]} purple stars for pitch {pitch}")
        
        # Debug: Show some purple star coordinates
        print(f"Sample purple star coordinates (first 10):")
        for i in range(min(10, len(top_coords))):
            print(f"  Purple star {i+1}: ({top_coords[i,1]}, {top_coords[i,0]}) - Score: {score_flat[top_indices[i]]:.2f}")
    
    # Overlay notes from MIDI as red circles
    try:
        notes, _ = midi2note(midi_path, binary_velocity=False)
        hop_size = model.hopSize
        fs = model.fs
        print(f"Total MIDI notes found: {len(notes)}")
        
        # Only consider notes with the specified pitch
        target_notes = [note for note in notes if note.pitch == pitch]
        print(f"MIDI notes with pitch {pitch}: {len(target_notes)}")
        
        midi_points_plotted = 0
        red_dot_coords = []
        for note in target_notes:
            onset_frame = int(note.onset * fs // hop_size)
            offset_frame = int(note.offset * fs // hop_size)
            print(f"  Note: onset={note.onset:.2f}s, offset={note.offset:.2f}s, onset_frame={onset_frame}, offset_frame={offset_frame}")
            
            # Only overlay if within the selected frame range
            if (onset_frame >= frame_start and onset_frame < frame_end and
                offset_frame >= frame_start and offset_frame < frame_end):
                x_coord = offset_frame - frame_start
                y_coord = onset_frame - frame_start
                plt.scatter(x_coord, y_coord, s=100, marker='o', color='red', edgecolors='white', linewidth=1, label=f'MIDI note {pitch}' if midi_points_plotted == 0 else None)
                red_dot_coords.append((x_coord, y_coord))
                midi_points_plotted += 1
                print(f"    Plotted at coordinates: ({x_coord}, {y_coord})")
            else:
                print(f"    Skipped - outside frame range [{frame_start}, {frame_end})")
        print(f"Plotted {midi_points_plotted} red dots for MIDI notes with pitch {pitch}")
        
        # Debug: Check if any purple stars are near red dots
        if len(red_dot_coords) > 0 and len(top_coords) > 0:
            print(f"\nChecking proximity of purple stars to red dots:")
            for i, (red_x, red_y) in enumerate(red_dot_coords):
                print(f"  Red dot {i+1} at ({red_x}, {red_y}):")
                nearby_stars = 0
                for j, (purple_x, purple_y) in enumerate(top_coords):
                    distance = np.sqrt((purple_x - red_x)**2 + (purple_y - red_y)**2)
                    if distance < 20:  # Within 20 frames
                        nearby_stars += 1
                        if nearby_stars <= 3:  # Show first 3 nearby stars
                            print(f"    Nearby purple star {j+1}: ({purple_x}, {purple_y}) - distance: {distance:.1f}")
                print(f"    Total nearby purple stars: {nearby_stars}")

        # Overlay Viterbi-decoded intervals as green dots
        if hasattr(model, 'processFramesBatch'):
            # Prepare a dummy batch for Viterbi decoding
            # Use the same audio segment as in the plot
            import torch
            n_frames = score_matrix.shape[0]
            # Create a dummy framesBatch for the selected range
            # (Assume the original audio is available as 'audio' and sr)
            # This is a simplification; ideally, use the same frames as in the plot
            # But for now, just decode the whole segment
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            window_size = model.windowSize
            hop_size = model.hopSize
            n_samples = audio_tensor.shape[-1]
            n_frames_total = (n_samples - window_size) // hop_size + 1
            frames = []
            for i in range(n_frames_total):
                start_idx = i * hop_size
                end_idx = start_idx + window_size
                frame = audio_tensor[:, :, start_idx:end_idx]
                frames.append(frame)
            frames_batch = torch.stack(frames, dim=2)  # [1, 1, n_frames, window_size]
            with torch.no_grad():
                crf, ctx = model.processFramesBatch(frames_batch)
                viterbi_paths = crf.decode()
            # viterbi_paths is a list of lists, one per pitch
            try:
                pitch_index = model.targetMIDIPitch.index(pitch)
                intervals = viterbi_paths[pitch_index]
                green_dot_coords = []
                for (onset_frame, offset_frame) in intervals:
                    if (onset_frame >= frame_start and onset_frame < frame_end and
                        offset_frame >= frame_start and offset_frame < frame_end):
                        x_coord = offset_frame - frame_start
                        y_coord = onset_frame - frame_start
                        plt.scatter(x_coord, y_coord, s=100, marker='x', color='green', linewidth=2, label='Viterbi path' if len(green_dot_coords) == 0 else None)
                        green_dot_coords.append((x_coord, y_coord))
                print(f"Plotted {len(green_dot_coords)} green dots for Viterbi-decoded intervals for pitch {pitch}")
            except Exception as e:
                print(f"Could not overlay Viterbi path: {e}")
        
    except Exception as e:
        print(f"Could not overlay MIDI notes: {e}")
    
    # Only show legend if there are any handles
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.colorbar(im, label='Score Value')
    plt.tight_layout()
    
    # Create a second subplot for log-scale visualization
    plt.figure(figsize=(8, 8))
    
    # Apply signed log transformation: sign(x) * log(1 + |x|)
    # This preserves the sign while compressing the scale
    score_matrix_log = np.sign(score_matrix) * np.log1p(np.abs(score_matrix))
    
    im_log = plt.imshow(score_matrix_log, aspect='auto', origin='upper', cmap='coolwarm')
    plt.title(f'Score Matrix (Log Scale) for Pitch {pitch}, Frames {frame_start}-{frame_end}')
    plt.xlabel('Frame')
    plt.ylabel('Frame')
    
    # Re-plot the overlays on the log-scale plot
    if top_coords.shape[0] > 0:
        plt.scatter(top_coords[:,1], top_coords[:,0], s=80, marker='*', color='purple', alpha=0.7, label=f'Top {n_values} values')
    
    # Re-plot MIDI notes
    if 'red_dot_coords' in locals() and len(red_dot_coords) > 0:
        for x_coord, y_coord in red_dot_coords:
            plt.scatter(x_coord, y_coord, s=100, marker='o', color='red', edgecolors='white', linewidth=1)
    
    # Re-plot Viterbi path
    if 'green_dot_coords' in locals() and len(green_dot_coords) > 0:
        for x_coord, y_coord in green_dot_coords:
            plt.scatter(x_coord, y_coord, s=100, marker='x', color='green', linewidth=2)
    
    # Only show legend if there are any handles
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.colorbar(im_log, label='Signed Log Score Value')
    plt.tight_layout()
    
    # Save both plots
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    log_save_path = save_path.replace('.png', '_log.png')
    plt.savefig(log_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved plots to {save_path} and {log_save_path}")

def plot_noise_matrix(S_skip_batch, model, frame_start, frame_end, midi_path, pitch, save_path=None, std_multiplier=1.5, top_n=100, audio=None):
    """Plot the noise matrix (transition scores) for a given pitch from frame_start to frame_end, overlaying the top N values, points above mean + std_multiplier*std, and the played notes as red circles. Optionally overlay Viterbi path using audio."""
    # Extract noise matrix for the specified pitch
    try:
        pitch_index = model.targetMIDIPitch.index(pitch)
    except ValueError:
        print(f"Pitch {pitch} not found in model.targetMIDIPitch!")
        return
    
    # Check the actual shape of S_skip_batch
    print(f"[DEBUG] S_skip_batch shape: {S_skip_batch.shape}")
    
    # Noise matrix has shape [T-1, nBatch*nPitch] where T is number of frames
    # For single batch, it's [T-1, nPitch], but it might be flattened differently
    if len(S_skip_batch.shape) == 2:
        if S_skip_batch.shape[1] == 1:
            # Single value for all pitches (flattened)
            noise_matrix = S_skip_batch[frame_start:frame_end-1, 0].cpu().numpy()
        else:
            # Multiple pitches
            noise_matrix = S_skip_batch[frame_start:frame_end-1, pitch_index].cpu().numpy()
    elif len(S_skip_batch.shape) == 3:
        # Shape [T-1, 1, nPitch] (frames, batch, pitch)
        noise_matrix = S_skip_batch[frame_start:frame_end-1, 0, pitch_index].cpu().numpy()
    else:
        print(f"Unexpected S_skip_batch shape: {S_skip_batch.shape}")
        return
    
    print(f"[DEBUG] Noise matrix shape: {noise_matrix.shape}")
    print(f"[DEBUG] Noise matrix stats: min={noise_matrix.min():.4f}, max={noise_matrix.max():.4f}, mean={noise_matrix.mean():.4f}, std={noise_matrix.std():.4f}")
    
    # Create single plot (linear scale only)
    plt.figure(figsize=(12, 6))
    
    # Linear scale plot
    plt.plot(range(len(noise_matrix)), noise_matrix, 'b-', linewidth=1, alpha=0.7)
    plt.title(f'Noise Matrix (Transition Scores) for Pitch {pitch}, Frames {frame_start}-{frame_end}')
    plt.xlabel('Frame')
    plt.ylabel('Noise Score')
    plt.grid(True, alpha=0.3)
    
    # Find top N values
    top_indices = np.argsort(noise_matrix)[-top_n:]
    top_values = noise_matrix[top_indices]
    plt.scatter(top_indices, top_values, s=80, marker='*', color='purple', alpha=0.7, label=f'Top {top_n} values')
    
    # Find values above threshold
    threshold = noise_matrix.mean() + std_multiplier * noise_matrix.std()
    above_threshold_indices = np.where(noise_matrix > threshold)[0]
    above_threshold_values = noise_matrix[above_threshold_indices]
    if len(above_threshold_indices) > 0:
        plt.scatter(above_threshold_indices, above_threshold_values, s=60, marker='o', color='orange', alpha=0.5, label=f'Scores > mean+{std_multiplier}*std')
    
    # Overlay notes from MIDI as red circles
    try:
        notes, _ = midi2note(midi_path, binary_velocity=False)
        hop_size = model.hopSize
        fs = model.fs
        print(f"Total MIDI notes found: {len(notes)}")
        
        # Only consider notes with the specified pitch
        target_notes = [note for note in notes if note.pitch == pitch]
        print(f"MIDI notes with pitch {pitch}: {len(target_notes)}")
        
        midi_points_plotted = 0
        for note in target_notes:
            onset_frame = int(note.onset * fs // hop_size)
            offset_frame = int(note.offset * fs // hop_size)
            
            # Check if onset is within the frame range
            if frame_start <= onset_frame < frame_end:
                # Plot onset as red dot
                if onset_frame - frame_start < len(noise_matrix):
                    plt.scatter(onset_frame - frame_start, noise_matrix[onset_frame - frame_start], 
                               s=100, marker='o', color='red', edgecolors='white', linewidth=1, 
                               label='MIDI onset' if midi_points_plotted == 0 else None)
                    midi_points_plotted += 1
                    print(f"  Note: onset={note.onset:.2f}s, offset={note.offset:.2f}s, onset_frame={onset_frame}, offset_frame={offset_frame}")
                    print(f"    Plotted onset at frame: {onset_frame - frame_start}")
        
        print(f"Plotted {midi_points_plotted} red dots for MIDI onsets with pitch {pitch}")
        
    except Exception as e:
        print(f"Error processing MIDI file: {e}")
    
    # Overlay Viterbi path if audio is provided
    if audio is not None:
        try:
            # Get Viterbi path for this pitch
            crf, ctx = model.processFramesBatch(audio.unsqueeze(0).unsqueeze(0))
            path = crf.decode()
            
            # Extract intervals for this specific pitch
            pitch_intervals = path[pitch_index]  # pitch_index corresponds to this pitch
            
            viterbi_points_plotted = 0
            for onset_frame, offset_frame in pitch_intervals:
                # Check if onset is within the frame range
                if frame_start <= onset_frame < frame_end:
                    if onset_frame - frame_start < len(noise_matrix):
                        plt.scatter(onset_frame - frame_start, noise_matrix[onset_frame - frame_start], 
                                   s=80, marker='x', color='green', alpha=0.8, 
                                   label='Viterbi onset' if viterbi_points_plotted == 0 else None)
                        viterbi_points_plotted += 1
            
            print(f"Plotted {viterbi_points_plotted} green x markers for Viterbi-decoded onsets for pitch {pitch}")
            
        except Exception as e:
            print(f"Error processing Viterbi path: {e}")
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = f'noise_{pitch}_{frame_start}_{frame_end}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved noise matrix plot to {save_path}")
    
    plt.show()

def main():
    """Main function to run the inner product score analysis"""
    print("TransKun Inner Product Score Analysis")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_random_seed(123)
    
    # Load configuration
    config_data, config_path = load_configuration()
    
    # Create model config
    config = create_model_config(config_data)
    
    # Initialize model
    print("\n=== INITIALIZING MODEL ===")
    try:
        model = TransKun(config)
        print(f"✓ Model initialized successfully")
        print(f"  Device: {model.getDevice()}")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return
    
    # Load pretrained weights
    model = load_pretrained_weights(model)
    
    # Analyze model components
    model = analyze_model_components(model)
    
    # Load audio file
    audio_path = "sample/gymnopedie1.mp3"
    audio_data = load_audio_file(audio_path)
    
    if audio_data is None:
        print("✗ Failed to load audio file")
        return
    
    audio, sr = audio_data
    
    # Analyze inner product score
    S_batch, S_skip_batch, ctx = analyze_inner_product_score(model, audio, sr)
    
    # Remove or comment out the old visualize_scores call to avoid index errors
    # visualize_scores(S_batch, S_skip_batch, model)
    
    print("\n" + "=" * 50)
    print("✓ Inner product score analysis completed!")
    print("=" * 50)
    # Plot F3 score matrix from frame 200 to 400
    if S_batch is not None:
        plot_score(S_batch, model, frame_start=0, frame_end=400, midi_path='gymnopedie1_transcribed.mid', pitch=66, audio=audio)
        plot_noise_matrix(S_skip_batch, model, frame_start=0, frame_end=400, midi_path='gymnopedie1_transcribed.mid', pitch=66, audio=audio)

if __name__ == "__main__":
    main() 