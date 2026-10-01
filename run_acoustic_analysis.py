"""
==============================================================================
Burmese Daily Dialogue Corpus (BDDC) — Rigorous Acoustic Analysis Pipeline
==============================================================================
Author: Speech & Audio ML Research Team
Date: October 2026
Objective:
  Perform a rigorous, empirical acoustic and prosodic analysis of the 13
  synthetic speakers in the Burmese Daily Dialogue Corpus (BDDC) to quantitatively
  demonstrate speaker acoustic separability and character divergence.

Key Deliverables:
  - 01_dataset_summary.csv
  - 02_f0_statistics.csv
  - 03_speaking_rate.csv
  - 04_duration_statistics.csv
  - 05_mfcc_statistics.csv
  - 06_speaker_embeddings.csv
  - 07_distance_matrix.csv
  - 9 publication-grade figures in analysis/figures/
==============================================================================
"""

import os
import sys
import re
import time
import math
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Ensure UTF-8 output encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# Set deterministic random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Output directories
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models', 'spkrec-ecapa-voxceleb')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plotting style configuration (publication-grade, neutral, academic)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# ==============================================================================
# 1. Linguistic & Syllable Segmentation Tools
# ==============================================================================
def count_burmese_syllables(text):
    """
    Standard Burmese Syllable Segmentation (Sylbreak onset rule).
    Syllable break occurs before any consonant [က-အ] or independent vowel [ဣ-ဪဿ]
    unless:
      1. Preceded by virama (\u1039, stacked consonant)
      2. Followed by asat (\u103A, final killer/coda)
    """
    if not isinstance(text, str):
        return 0
    clean = re.sub(r'[\s၊။\.\,\?\!\:\;\"\'\(\)\-\_]+', '', text)
    if not clean:
        return 0
    pattern = re.compile(r'(?<!\u1039)([က-အဣ-ဪဿ])(?!\u103A)')
    matches = list(pattern.finditer(clean))
    return max(1, len(matches))

# ==============================================================================
# 2. Worker Function for Audio Feature Extraction
# ==============================================================================
def process_single_utterance(row_tuple):
    """
    Worker function executed in parallel across CPU cores.
    Extracts:
      - Total duration, active speech duration (VAD), pause duration, active ratio
      - Syllables, characters, speaking rates (chars/s, syll/s)
      - F0 trajectory statistics (median, mean, std, p10, p90, range, voiced ratio)
      - MFCCs (13 means + 13 standard deviations)
    """
    idx, file_name, sentence, speaker_name, speaker_gender, split, reported_dur = row_tuple
    
    try:
        y, sr = sf.read(file_name)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        total_dur = len(y) / sr
        clean_text = re.sub(r'[\s၊။\.\,\?\!\:\;\"\'\(\)\-\_]+', '', str(sentence))
        n_chars = max(1, len(clean_text))
        n_sylls = max(1, count_burmese_syllables(sentence))
        
        # Energy VAD (25ms window, 10ms hop)
        win_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        n_frames = max(1, (len(y) - win_len) // hop_len + 1)
        
        if len(y) >= win_len:
            shape = (n_frames, win_len)
            strides = (y.strides[0] * hop_len, y.strides[0])
            frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
            rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
            threshold = max(0.005, np.percentile(rms, 95) * 0.1)
            active_mask = rms > threshold
            active_frames = np.sum(active_mask)
            active_dur = active_frames * hop_len / sr
        else:
            active_dur = total_dur
            
        pause_dur = max(0.0, total_dur - active_dur)
        active_ratio = active_dur / max(0.001, total_dur)
        
        chars_per_sec = n_chars / max(0.01, total_dur)
        chars_per_sec_active = n_chars / max(0.01, active_dur)
        syll_per_sec = n_sylls / max(0.01, total_dur)
        syll_per_sec_active = n_sylls / max(0.01, active_dur)
        
        sec_per_char = total_dur / n_chars
        sec_per_syll = total_dur / n_sylls
        
        # F0 Autocorrelation (30ms window, 10ms hop, range [60, 500] Hz)
        f0_win = int(0.030 * sr)
        min_lag = int(sr / 500)
        max_lag = int(sr / 60)
        
        f0_n_frames = max(1, (len(y) - f0_win) // hop_len + 1)
        f0_list = []
        
        if len(y) >= f0_win:
            shape_f0 = (f0_n_frames, f0_win)
            strides_f0 = (y.strides[0] * hop_len, y.strides[0])
            f0_frames = np.lib.stride_tricks.as_strided(y, shape=shape_f0, strides=strides_f0)
            hanning = np.hanning(f0_win)
            windowed = f0_frames * hanning
            
            n_fft = 1 << (2 * f0_win - 1).bit_length()
            fx = np.fft.rfft(windowed, n=n_fft, axis=1)
            r = np.fft.irfft(fx * np.conj(fx), n=n_fft, axis=1)[:, :f0_win]
            
            energy = r[:, 0:1]
            energy = np.where(energy <= 0, 1.0, energy)
            norm_r = r / energy
            
            lag_sub = norm_r[:, min_lag:max_lag]
            best_lag_rel = np.argmax(lag_sub, axis=1)
            best_lag = best_lag_rel + min_lag
            peak_vals = np.take_along_axis(norm_r, best_lag[:, None], axis=1).squeeze(1)
            
            for f_idx in range(f0_n_frames):
                val = peak_vals[f_idx]
                k = best_lag[f_idx]
                if val > 0.45 and k > min_lag and k < max_lag - 1:
                    alpha = norm_r[f_idx, k - 1]
                    beta = norm_r[f_idx, k]
                    gamma = norm_r[f_idx, k + 1]
                    denom = 2 * (2 * beta - alpha - gamma)
                    if abs(denom) > 1e-6:
                        delta = (gamma - alpha) / denom
                        k_refined = k + delta
                        f0 = sr / k_refined
                        if 60 <= f0 <= 500:
                            f0_list.append(f0)
                            
        voiced_count = len(f0_list)
        voiced_ratio = voiced_count / max(1, f0_n_frames)
        
        if voiced_count >= 5:
            f0_arr = np.array(f0_list)
            f0_median = float(np.median(f0_arr))
            f0_mean = float(np.mean(f0_arr))
            f0_std = float(np.std(f0_arr))
            f0_p10 = float(np.percentile(f0_arr, 10))
            f0_p90 = float(np.percentile(f0_arr, 90))
            f0_range = f0_p90 - f0_p10
        else:
            f0_median = np.nan
            f0_mean = np.nan
            f0_std = np.nan
            f0_p10 = np.nan
            f0_p90 = np.nan
            f0_range = np.nan
            
        # MFCC extraction (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=160, win_length=400)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        record = {
            'id': idx,
            'file_name': file_name,
            'speaker_name': speaker_name,
            'speaker_gender': speaker_gender,
            'split': split,
            'total_duration': total_dur,
            'active_speech_duration': active_dur,
            'pause_duration': pause_dur,
            'active_ratio': active_ratio,
            'n_chars': n_chars,
            'n_syllables': n_sylls,
            'chars_per_sec': chars_per_sec,
            'chars_per_sec_active': chars_per_sec_active,
            'syll_per_sec': syll_per_sec,
            'syll_per_sec_active': syll_per_sec_active,
            'sec_per_char': sec_per_char,
            'sec_per_syll': sec_per_syll,
            'f0_median': f0_median,
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_p10': f0_p10,
            'f0_p90': f0_p90,
            'f0_range': f0_range,
            'voiced_ratio': voiced_ratio
        }
        for m in range(13):
            record[f'mfcc_mean_{m+1}'] = float(mfcc_mean[m])
            record[f'mfcc_std_{m+1}'] = float(mfcc_std[m])
            
        return record
    except Exception as e:
        return {'id': idx, 'error': str(e)}

# ==============================================================================
# 3. Main Acoustic Analysis Execution
# ==============================================================================
def main():
    print("=" * 80)
    print("🚀 STARTING BURMESE DAILY DIALOGUE CORPUS (BDDC) ACOUSTIC ANALYSIS")
    print("=" * 80)
    start_time = time.time()
    
    # 1. Load dataset metadata
    csv_path = 'all_data_sequential.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
        
    print(f"📖 Reading dataset metadata from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    total_utterances = len(df_raw)
    print(f"   Total corpus rows: {total_utterances:,}")
    
    # Generate 01_dataset_summary.csv
    print("\n📊 1. Generating Dataset Summary (01_dataset_summary.csv)...")
    summary_rows = []
    speakers = sorted(df_raw['speaker_name'].unique())
    
    for spk in speakers:
        sub = df_raw[df_raw['speaker_name'] == spk]
        gender = sub['speaker_gender'].iloc[0]
        n_utt = len(sub)
        tot_dur_sec = sub['duration'].sum()
        tot_dur_hr = tot_dur_sec / 3600.0
        mean_dur = sub['duration'].mean()
        median_dur = sub['duration'].median()
        std_dur = sub['duration'].std()
        min_dur = sub['duration'].min()
        max_dur = sub['duration'].max()
        
        train_c = len(sub[sub['split'] == 'train'])
        val_c = len(sub[sub['split'] == 'val'])
        test_c = len(sub[sub['split'] == 'test'])
        is_test_only = (train_c == 0 and val_c == 0 and test_c > 0)
        
        summary_rows.append({
            'speaker_name': spk,
            'speaker_gender': gender,
            'total_utterances': n_utt,
            'total_duration_sec': round(tot_dur_sec, 2),
            'total_duration_hours': round(tot_dur_hr, 4),
            'mean_duration_sec': round(mean_dur, 3),
            'median_duration_sec': round(median_dur, 3),
            'std_duration_sec': round(std_dur, 3),
            'min_duration_sec': round(min_dur, 3),
            'max_duration_sec': round(max_dur, 3),
            'train_count': train_c,
            'val_count': val_c,
            'test_count': test_c,
            'is_test_only': is_test_only
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary_path = os.path.join(OUTPUT_DIR, '01_dataset_summary.csv')
    df_summary.to_csv(df_summary_path, index=False)
    print(f"   Saved dataset summary: {df_summary_path}")
    print(df_summary[['speaker_name', 'speaker_gender', 'total_utterances', 'total_duration_hours', 'median_duration_sec', 'is_test_only']].to_string(index=False))

    # 2. Parallel Audio Feature Extraction
    print("\n🎙️ 2. Parallel Extraction: F0, Speaking Rates, Durations, and MFCCs (Full Corpus)...")
    tasks = []
    for idx, row in df_raw.iterrows():
        tasks.append((row['id'], row['file_name'], row['sentence'], row['speaker_name'], row['speaker_gender'], row['split'], row['duration']))
        
    num_workers = min(10, os.cpu_count() or 4)
    print(f"   Running parallel extraction using {num_workers} CPU worker processes...")
    t_feat_start = time.time()
    
    extracted_records = []
    # Process in chunks using ProcessPoolExecutor
    chunk_size = 500
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            results = list(executor.map(process_single_utterance, chunk))
            for res in results:
                if 'error' not in res:
                    extracted_records.append(res)
            sys.stdout.write(f"\r   Processed {len(extracted_records):,}/{len(tasks):,} utterances ({len(extracted_records)/len(tasks)*100:.1f}%)")
            sys.stdout.flush()
            
    print(f"\n   Feature extraction completed in {time.time() - t_feat_start:.2f} seconds.")
    df_features = pd.DataFrame(extracted_records)
    
    # Save intermediate complete utterance-level cache
    feature_cache_path = os.path.join(OUTPUT_DIR, 'utterance_features_full.csv')
    df_features.to_csv(feature_cache_path, index=False)
    print(f"   Cached utterance features to {feature_cache_path}")
    
    # 3. F0 Statistics Generation (02_f0_statistics.csv)
    print("\n🎼 3. Generating F0 Statistics (02_f0_statistics.csv)...")
    f0_rows = []
    for spk in speakers:
        sub = df_features[df_features['speaker_name'] == spk].dropna(subset=['f0_median'])
        gender = sub['speaker_gender'].iloc[0]
        f0_rows.append({
            'speaker_name': spk,
            'speaker_gender': gender,
            'median_f0_hz': round(sub['f0_median'].median(), 2),
            'mean_f0_hz': round(sub['f0_mean'].mean(), 2),
            'std_f0_hz': round(sub['f0_std'].mean(), 2),
            'p10_f0_hz': round(sub['f0_p10'].median(), 2),
            'p90_f0_hz': round(sub['f0_p90'].median(), 2),
            'f0_range_hz': round(sub['f0_range'].median(), 2),
            'voiced_ratio_mean': round(sub['voiced_ratio'].mean(), 3),
            'voiced_ratio_std': round(sub['voiced_ratio'].std(), 3),
            'valid_voiced_utterances': len(sub)
        })
    df_f0 = pd.DataFrame(f0_rows)
    df_f0_path = os.path.join(OUTPUT_DIR, '02_f0_statistics.csv')
    df_f0.to_csv(df_f0_path, index=False)
    print(f"   Saved F0 statistics: {df_f0_path}")
    print(df_f0[['speaker_name', 'speaker_gender', 'median_f0_hz', 'f0_range_hz', 'voiced_ratio_mean']].to_string(index=False))

    # 4. Speaking Rate Statistics (03_speaking_rate.csv)
    print("\n⏱️ 4. Generating Speaking Rate Statistics (03_speaking_rate.csv)...")
    rate_rows = []
    for spk in speakers:
        sub = df_features[df_features['speaker_name'] == spk]
        gender = sub['speaker_gender'].iloc[0]
        
        syll_median = sub['syll_per_sec'].median()
        syll_mean = sub['syll_per_sec'].mean()
        syll_iqr = sub['syll_per_sec'].quantile(0.75) - sub['syll_per_sec'].quantile(0.25)
        syll_std = sub['syll_per_sec'].std()
        
        syll_act_median = sub['syll_per_sec_active'].median()
        syll_act_mean = sub['syll_per_sec_active'].mean()
        
        char_median = sub['chars_per_sec'].median()
        char_mean = sub['chars_per_sec'].mean()
        char_iqr = sub['chars_per_sec'].quantile(0.75) - sub['chars_per_sec'].quantile(0.25)
        char_std = sub['chars_per_sec'].std()
        
        rate_rows.append({
            'speaker_name': spk,
            'speaker_gender': gender,
            'syll_per_sec_median': round(syll_median, 3),
            'syll_per_sec_mean': round(syll_mean, 3),
            'syll_per_sec_iqr': round(syll_iqr, 3),
            'syll_per_sec_std': round(syll_std, 3),
            'syll_per_sec_active_median': round(syll_act_median, 3),
            'syll_per_sec_active_mean': round(syll_act_mean, 3),
            'chars_per_sec_median': round(char_median, 3),
            'chars_per_sec_mean': round(char_mean, 3),
            'chars_per_sec_iqr': round(char_iqr, 3),
            'chars_per_sec_std': round(char_std, 3)
        })
    df_rate = pd.DataFrame(rate_rows)
    df_rate_path = os.path.join(OUTPUT_DIR, '03_speaking_rate.csv')
    df_rate.to_csv(df_rate_path, index=False)
    print(f"   Saved speaking rate statistics: {df_rate_path}")
    print(df_rate[['speaker_name', 'speaker_gender', 'syll_per_sec_median', 'chars_per_sec_median', 'syll_per_sec_std']].to_string(index=False))

    # 5. Duration Statistics (04_duration_statistics.csv)
    print("\n⏳ 5. Generating Duration Statistics (04_duration_statistics.csv)...")
    dur_rows = []
    for spk in speakers:
        sub = df_features[df_features['speaker_name'] == spk]
        gender = sub['speaker_gender'].iloc[0]
        
        dur_rows.append({
            'speaker_name': spk,
            'speaker_gender': gender,
            'mean_total_dur_sec': round(sub['total_duration'].mean(), 3),
            'median_total_dur_sec': round(sub['total_duration'].median(), 3),
            'mean_active_dur_sec': round(sub['active_speech_duration'].mean(), 3),
            'median_active_dur_sec': round(sub['active_speech_duration'].median(), 3),
            'mean_pause_dur_sec': round(sub['pause_duration'].mean(), 3),
            'median_pause_dur_sec': round(sub['pause_duration'].median(), 3),
            'sec_per_char_median': round(sub['sec_per_char'].median(), 4),
            'sec_per_char_mean': round(sub['sec_per_char'].mean(), 4),
            'sec_per_syll_median': round(sub['sec_per_syll'].median(), 4),
            'sec_per_syll_mean': round(sub['sec_per_syll'].mean(), 4),
            'active_ratio_mean': round(sub['active_ratio'].mean(), 3)
        })
    df_dur = pd.DataFrame(dur_rows)
    df_dur_path = os.path.join(OUTPUT_DIR, '04_duration_statistics.csv')
    df_dur.to_csv(df_dur_path, index=False)
    print(f"   Saved duration statistics: {df_dur_path}")
    print(df_dur[['speaker_name', 'speaker_gender', 'mean_total_dur_sec', 'sec_per_syll_median', 'active_ratio_mean']].to_string(index=False))

    # 6. MFCC Statistics (05_mfcc_statistics.csv)
    print("\n🎛️ 6. Generating MFCC Spectral Statistics (05_mfcc_statistics.csv)...")
    mfcc_rows = []
    mfcc_cols_mean = [f'mfcc_mean_{i}' for i in range(1, 14)]
    mfcc_cols_std = [f'mfcc_std_{i}' for i in range(1, 14)]
    
    for spk in speakers:
        sub = df_features[df_features['speaker_name'] == spk]
        gender = sub['speaker_gender'].iloc[0]
        row_dict = {'speaker_name': spk, 'speaker_gender': gender}
        for col in mfcc_cols_mean + mfcc_cols_std:
            row_dict[col] = round(sub[col].mean(), 4)
        mfcc_rows.append(row_dict)
        
    df_mfcc = pd.DataFrame(mfcc_rows)
    df_mfcc_path = os.path.join(OUTPUT_DIR, '05_mfcc_statistics.csv')
    df_mfcc.to_csv(df_mfcc_path, index=False)
    print(f"   Saved MFCC statistics: {df_mfcc_path}")

    # 7. Pretrained ECAPA-TDNN Speaker Embeddings (06_speaker_embeddings.csv)
    print("\n🧠 7. Extracting Pretrained ECAPA-TDNN Speaker Embeddings (06_speaker_embeddings.csv)...")
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source=MODELS_DIR,
        run_opts={'device': 'cpu'},
        savedir=MODELS_DIR
    )
    
    # Stratified balanced sampling: 100 representative utterances per speaker
    # (Pasada has 398; 100 provides a representative, non-redundant sample)
    SAMPLES_PER_SPEAKER = 100
    spk_embeddings = {}
    
    for spk in speakers:
        sub = df_raw[df_raw['speaker_name'] == spk]
        # Stratified sample with deterministic seed
        sample_indices = sub.sample(n=min(SAMPLES_PER_SPEAKER, len(sub)), random_state=SEED).index
        audio_paths = df_raw.loc[sample_indices, 'file_name'].tolist()
        
        embs_list = []
        for a_path in audio_paths:
            sig, sr_audio = sf.read(a_path)
            if len(sig.shape) > 1:
                sig = sig.mean(axis=1)
            sig_t = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = classifier.encode_batch(sig_t) # shape (1, 1, 192)
                emb_np = emb.squeeze().cpu().numpy()
                # Unit normalize individual embedding
                emb_norm = emb_np / (np.linalg.norm(emb_np) + 1e-12)
                embs_list.append(emb_norm)
                
        # Centroid embedding for the speaker
        centroid = np.mean(embs_list, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        spk_embeddings[spk] = centroid
        print(f"   Extracted {len(embs_list)} ECAPA-TDNN embeddings for speaker: {spk:<8}")
        
    emb_rows = []
    for spk in speakers:
        row_dict = {'speaker_name': spk}
        for d in range(192):
            row_dict[f'emb_{d}'] = float(spk_embeddings[spk][d])
        emb_rows.append(row_dict)
        
    df_embeddings = pd.DataFrame(emb_rows)
    df_embeddings_path = os.path.join(OUTPUT_DIR, '06_speaker_embeddings.csv')
    df_embeddings.to_csv(df_embeddings_path, index=False)
    print(f"   Saved speaker embeddings: {df_embeddings_path}")

    # 8. Pairwise Distances & Combined Acoustic Space (07_distance_matrix.csv)
    print("\n📐 8. Computing Distance Matrices & Combined Acoustic Space...")
    
    # 8a. Embedding Distance Matrix (Cosine Distance)
    emb_matrix = np.array([spk_embeddings[spk] for spk in speakers])
    dist_emb = squareform(pdist(emb_matrix, metric='cosine'))
    
    # 8b. MFCC Distance Matrix (Euclidean on standardized speaker-level MFCCs)
    mfcc_feature_cols = mfcc_cols_mean + mfcc_cols_std
    mfcc_raw_matrix = df_mfcc[mfcc_feature_cols].values
    scaler_mfcc = StandardScaler()
    mfcc_norm_matrix = scaler_mfcc.fit_transform(mfcc_raw_matrix)
    dist_mfcc = squareform(pdist(mfcc_norm_matrix, metric='cosine'))
    
    # 8c. F0 & Prosodic Distance Matrix
    f0_prosody_features = ['median_f0_hz', 'f0_range_hz', 'voiced_ratio_mean']
    f0_raw_matrix = df_f0[f0_prosody_features].values
    scaler_f0 = StandardScaler()
    f0_norm_matrix = scaler_f0.fit_transform(f0_raw_matrix)
    dist_f0 = squareform(pdist(f0_norm_matrix, metric='euclidean'))
    dist_f0 = dist_f0 / (np.max(dist_f0) + 1e-12) # normalize to [0, 1]
    
    # 8d. Speaking Rate & Duration Style Distance Matrix
    style_features = ['syll_per_sec_median', 'chars_per_sec_median', 'sec_per_syll_median', 'active_ratio_mean']
    style_df_combined = pd.merge(df_rate[['speaker_name', 'syll_per_sec_median', 'chars_per_sec_median']],
                                 df_dur[['speaker_name', 'sec_per_syll_median', 'active_ratio_mean']],
                                 on='speaker_name')
    style_raw_matrix = style_df_combined[style_features].values
    scaler_style = StandardScaler()
    style_norm_matrix = scaler_style.fit_transform(style_raw_matrix)
    dist_style = squareform(pdist(style_norm_matrix, metric='euclidean'))
    dist_style = dist_style / (np.max(dist_style) + 1e-12)
    
    # Combined Acoustic Distance: Balanced multi-view combination
    # Voice Identity (ECAPA + MFCC) = 60%, Pitch/Prosody (F0) = 20%, Rhythm/Style (Rate) = 20%
    # This prevents the 192-dim space from dominating while reflecting perceptual acoustic distance.
    combined_dist = (0.35 * dist_emb) + (0.25 * dist_mfcc) + (0.20 * dist_f0) + (0.20 * dist_style)
    # Ensure diagonal is strictly zero
    np.fill_diagonal(combined_dist, 0.0)
    
    df_combined_dist = pd.DataFrame(combined_dist, index=speakers, columns=speakers)
    df_combined_dist_path = os.path.join(OUTPUT_DIR, '07_distance_matrix.csv')
    df_combined_dist.to_csv(df_combined_dist_path)
    print(f"   Saved combined distance matrix: {df_combined_dist_path}")

    # ==============================================================================
    # 9. Publication-Grade Figure Generation
    # ==============================================================================
    print("\n🎨 9. Generating Publication-Grade Figures (Figures 1 to 9)...")
    
    # Palette definition (Neutral, accessible, colorblind-friendly)
    male_color = '#1f77b4'
    female_color = '#d62728'
    spk_genders = {row['speaker_name']: row['speaker_gender'] for _, row in df_summary.iterrows()}
    palette = {s: (female_color if spk_genders[s] == 'female' else male_color) for s in speakers}

    # --------------------------------------------------------------------------
    # Figure 1: F0 distribution by speaker
    # --------------------------------------------------------------------------
    print("   Plotting Figure 1: f0_by_speaker.png...")
    plt.figure(figsize=(12, 6))
    df_f0_plot = df_features.dropna(subset=['f0_median'])
    # Sort speakers by median F0
    sorted_spk_f0 = df_f0.sort_values('median_f0_hz')['speaker_name'].tolist()
    
    sns.boxplot(
        data=df_f0_plot,
        x='speaker_name',
        y='f0_median',
        order=sorted_spk_f0,
        palette=palette,
        fliersize=1,
        linewidth=1.2,
        boxprops=dict(alpha=0.8)
    )
    plt.axhline(165, color='gray', linestyle=':', label='Gender Ambiguity / Pitch Transition (~165 Hz)')
    plt.title('Figure 1: Fundamental Frequency (F0) Distribution Across BDDC Speakers', pad=15)
    plt.xlabel('Speaker (Ordered by Median F0)')
    plt.ylabel('Median F0 per Utterance (Hz)')
    plt.ylim(60, 360)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'f0_by_speaker.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 2: Speaking rate distribution by speaker
    # --------------------------------------------------------------------------
    print("   Plotting Figure 2: speaking_rate_by_speaker.png...")
    plt.figure(figsize=(12, 6))
    sorted_spk_rate = df_rate.sort_values('syll_per_sec_median')['speaker_name'].tolist()
    sns.boxplot(
        data=df_features,
        x='speaker_name',
        y='syll_per_sec',
        order=sorted_spk_rate,
        palette=palette,
        fliersize=1,
        linewidth=1.2,
        boxprops=dict(alpha=0.8)
    )
    plt.title('Figure 2: Syllable Speaking Rate Distribution Across BDDC Speakers', pad=15)
    plt.xlabel('Speaker (Ordered by Median Syllable Rate)')
    plt.ylabel('Speaking Rate (Syllables / Second)')
    plt.ylim(1.5, 7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'speaking_rate_by_speaker.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 3: Speaking rate vs Utterance duration
    # --------------------------------------------------------------------------
    print("   Plotting Figure 3: rate_vs_duration.png...")
    plt.figure(figsize=(10, 6))
    # Downsample for scatter visibility if needed (e.g. 5,000 points)
    sample_df = df_features.sample(n=min(5000, len(df_features)), random_state=SEED)
    sns.scatterplot(
        data=sample_df,
        x='total_duration',
        y='syll_per_sec',
        hue='speaker_name',
        alpha=0.4,
        s=20,
        palette='tab20'
    )
    plt.title('Figure 3: Syllable Speaking Rate vs. Total Utterance Duration', pad=15)
    plt.xlabel('Total Utterance Duration (Seconds)')
    plt.ylabel('Speaking Rate (Syllables / Second)')
    plt.xlim(0.5, 12.0)
    plt.ylim(1.5, 8.0)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title='Speaker', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rate_vs_duration.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 4: Normalized duration by speaker
    # --------------------------------------------------------------------------
    print("   Plotting Figure 4: duration_by_speaker.png...")
    plt.figure(figsize=(12, 6))
    sorted_spk_dur = df_dur.sort_values('sec_per_syll_median')['speaker_name'].tolist()
    sns.boxplot(
        data=df_features,
        x='speaker_name',
        y='sec_per_syll',
        order=sorted_spk_dur,
        palette=palette,
        fliersize=1,
        linewidth=1.2,
        boxprops=dict(alpha=0.8)
    )
    plt.title('Figure 4: Normalized Acoustic Duration (Seconds per Syllable) Across Speakers', pad=15)
    plt.xlabel('Speaker (Ordered by Median Seconds/Syllable)')
    plt.ylabel('Normalized Duration (Seconds / Syllable)')
    plt.ylim(0.10, 0.55)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'duration_by_speaker.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 5: MFCC PCA Projection
    # --------------------------------------------------------------------------
    print("   Plotting Figure 5: mfcc_pca.png...")
    pca_mfcc = PCA(n_components=2, random_state=SEED)
    mfcc_pca_pts = pca_mfcc.fit_transform(mfcc_norm_matrix)
    
    plt.figure(figsize=(9, 7))
    for i, spk in enumerate(speakers):
        col = female_color if spk_genders[spk] == 'female' else male_color
        marker = 'o' if spk_genders[spk] == 'female' else 's'
        plt.scatter(mfcc_pca_pts[i, 0], mfcc_pca_pts[i, 1], color=col, marker=marker, s=120, edgecolors='black', linewidth=1.2, zorder=3)
        plt.annotate(
            f" {spk}",
            (mfcc_pca_pts[i, 0], mfcc_pca_pts[i, 1]),
            fontsize=11,
            fontweight='bold',
            va='center',
            ha='left' if mfcc_pca_pts[i, 0] >= 0 else 'right'
        )
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
    plt.title(f'Figure 5: 2D PCA of Speaker MFCC Spectral Centroids\n(PC1: {pca_mfcc.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca_mfcc.explained_variance_ratio_[1]*100:.1f}%)', pad=15)
    plt.xlabel(f'Principal Component 1 ({pca_mfcc.explained_variance_ratio_[0]*100:.1f}% var)')
    plt.ylabel(f'Principal Component 2 ({pca_mfcc.explained_variance_ratio_[1]*100:.1f}% var)')
    # Custom legend for gender
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Female Speaker', markerfacecolor=female_color, markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Male Speaker', markerfacecolor=male_color, markersize=10)
    ]
    plt.legend(handles=legend_elements, frameon=True, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'mfcc_pca.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 6: Embedding PCA Projection
    # --------------------------------------------------------------------------
    print("   Plotting Figure 6: embedding_pca.png...")
    pca_emb = PCA(n_components=2, random_state=SEED)
    emb_pca_pts = pca_emb.fit_transform(emb_matrix)
    
    plt.figure(figsize=(9, 7))
    for i, spk in enumerate(speakers):
        col = female_color if spk_genders[spk] == 'female' else male_color
        marker = 'o' if spk_genders[spk] == 'female' else 's'
        plt.scatter(emb_pca_pts[i, 0], emb_pca_pts[i, 1], color=col, marker=marker, s=120, edgecolors='black', linewidth=1.2, zorder=3)
        plt.annotate(
            f" {spk}",
            (emb_pca_pts[i, 0], emb_pca_pts[i, 1]),
            fontsize=11,
            fontweight='bold',
            va='center',
            ha='left' if emb_pca_pts[i, 0] >= 0 else 'right'
        )
    plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(0, color='gray', linestyle=':', alpha=0.5)
    plt.title(f'Figure 6: 2D PCA of ECAPA-TDNN Pretrained Speaker Embeddings\n(PC1: {pca_emb.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca_emb.explained_variance_ratio_[1]*100:.1f}%)', pad=15)
    plt.xlabel(f'Principal Component 1 ({pca_emb.explained_variance_ratio_[0]*100:.1f}% var)')
    plt.ylabel(f'Principal Component 2 ({pca_emb.explained_variance_ratio_[1]*100:.1f}% var)')
    plt.legend(handles=legend_elements, frameon=True, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'embedding_pca.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 7: ECAPA-TDNN Cosine Distance Heatmap
    # --------------------------------------------------------------------------
    print("   Plotting Figure 7: embedding_heatmap.png...")
    plt.figure(figsize=(10, 8.5))
    df_dist_emb = pd.DataFrame(dist_emb, index=speakers, columns=speakers)
    sns.heatmap(
        df_dist_emb,
        annot=True,
        fmt='.2f',
        cmap='viridis_r',
        vmin=0.0,
        vmax=float(np.max(dist_emb)),
        cbar_kws={'label': 'Cosine Distance (0.0 = Identical)'},
        linewidths=0.5,
        square=True
    )
    plt.title('Figure 7: Pairwise Cosine Distance Heatmap (ECAPA-TDNN Embeddings)', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'embedding_heatmap.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 8: Combined Distance Heatmap
    # --------------------------------------------------------------------------
    print("   Plotting Figure 8: combined_distance_heatmap.png...")
    plt.figure(figsize=(10, 8.5))
    sns.heatmap(
        df_combined_dist,
        annot=True,
        fmt='.2f',
        cmap='rocket_r',
        vmin=0.0,
        vmax=float(np.max(combined_dist)),
        cbar_kws={'label': 'Combined Acoustic Distance'},
        linewidths=0.5,
        square=True
    )
    plt.title('Figure 8: Combined Multi-View Acoustic Distance Heatmap', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'combined_distance_heatmap.png'))
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 9: Hierarchical Clustering Dendrogram
    # --------------------------------------------------------------------------
    print("   Plotting Figure 9: speaker_dendrogram.png...")
    plt.figure(figsize=(11, 6))
    condensed_dist = squareform(combined_dist)
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    # Calculate optimal threshold for 2-4 major clusters
    dendrogram(
        linkage_matrix,
        labels=speakers,
        leaf_rotation=0,
        leaf_font_size=11,
        color_threshold=0.5 * max(linkage_matrix[:, 2])
    )
    plt.title("Figure 9: Ward's Hierarchical Clustering Dendrogram Across 13 BDDC Speakers", pad=15)
    plt.xlabel('Speaker Identity')
    plt.ylabel('Ward Cluster Cophenetic Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'speaker_dendrogram.png'))
    plt.close()

    # ==============================================================================
    # 10. Statistical Hypothesis Testing & Effect Sizes
    # ==============================================================================
    print("\n🔬 10. Performing Statistical Tests (Kruskal-Wallis & Effect Sizes)...")
    
    # F0 Kruskal-Wallis
    f0_groups = [group['f0_median'].dropna().values for _, group in df_features.groupby('speaker_name')]
    h_f0, p_f0 = stats.kruskal(*f0_groups)
    N_f0 = sum(len(g) for g in f0_groups)
    k_f0 = len(f0_groups)
    eta2_f0 = (h_f0 - k_f0 + 1) / (N_f0 - k_f0)
    
    # Speaking Rate Kruskal-Wallis
    rate_groups = [group['syll_per_sec'].dropna().values for _, group in df_features.groupby('speaker_name')]
    h_rate, p_rate = stats.kruskal(*rate_groups)
    N_rate = sum(len(g) for g in rate_groups)
    k_rate = len(rate_groups)
    eta2_rate = (h_rate - k_rate + 1) / (N_rate - k_rate)
    
    # Normalized Duration Kruskal-Wallis
    dur_groups = [group['sec_per_syll'].dropna().values for _, group in df_features.groupby('speaker_name')]
    h_dur, p_dur = stats.kruskal(*dur_groups)
    N_dur = sum(len(g) for g in dur_groups)
    k_dur = len(dur_groups)
    eta2_dur = (h_dur - k_dur + 1) / (N_dur - k_dur)
    
    print(f"   F0 Kruskal-Wallis:         H = {h_f0:.1f}, p = {p_f0:.2e}, eta^2 = {eta2_f0:.4f}")
    print(f"   Speaking Rate H-test:      H = {h_rate:.1f}, p = {p_rate:.2e}, eta^2 = {eta2_rate:.4f}")
    print(f"   Sec/Syllable H-test:       H = {h_dur:.1f}, p = {p_dur:.2e}, eta^2 = {eta2_dur:.4f}")

    # ==============================================================================
    # 11. Table 1 & Table 2 Generation
    # ==============================================================================
    print("\n📋 11. Building Final Publication Tables...")
    
    # Hierarchical Cluster Assignments (k=3 or 4)
    cluster_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')
    spk_cluster_map = {speakers[i]: f"Cluster {cluster_labels[i]}" for i in range(len(speakers))}
    
    # MFCC PC1-based cluster/partition
    mfcc_pc1 = mfcc_pca_pts[:, 0]
    mfcc_cluster_map = {speakers[i]: ("Spectral Group A" if mfcc_pc1[i] >= 0 else "Spectral Group B") for i in range(len(speakers))}
    
    # Table 1: Concise Speaker Summary
    table1_rows = []
    for spk in speakers:
        f0_s = df_f0[df_f0['speaker_name'] == spk].iloc[0]
        r_s = df_rate[df_rate['speaker_name'] == spk].iloc[0]
        d_s = df_dur[df_dur['speaker_name'] == spk].iloc[0]
        sum_s = df_summary[df_summary['speaker_name'] == spk].iloc[0]
        
        table1_rows.append({
            'Speaker': spk,
            'Voice Type': f"{sum_s['speaker_gender'].capitalize()} {'(Test)' if sum_s['is_test_only'] else ''}",
            'Median F0 (Hz)': f0_s['median_f0_hz'],
            'F0 Range (Hz)': f0_s['f0_range_hz'],
            'Speech Rate (syll/s)': r_s['syll_per_sec_median'],
            'Median Duration (s)': d_s['median_total_dur_sec'],
            'MFCC Position': mfcc_cluster_map[spk],
            'Embedding Cluster': spk_cluster_map[spk]
        })
    df_table1 = pd.DataFrame(table1_rows)
    table1_path = os.path.join(OUTPUT_DIR, 'table1_speaker_summary.csv')
    df_table1.to_csv(table1_path, index=False)
    print(f"   Table 1 saved to: {table1_path}")
    print(df_table1.to_string(index=False))

    # Table 2: Pairwise Distance Analysis (Extract closest and most distant pairs)
    pairs_list = []
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            spk_a = speakers[i]
            spk_b = speakers[j]
            c_dist = combined_dist[i, j]
            e_dist = dist_emb[i, j]
            m_dist = dist_mfcc[i, j]
            f_dist = dist_f0[i, j]
            
            # Determine primary separating evidence
            evidences = []
            if e_dist > 0.4:
                evidences.append('ECAPA-TDNN Embedding')
            if m_dist > 0.3:
                evidences.append('MFCC Spectral')
            if f_dist > 0.3:
                evidences.append('F0 Pitch')
            main_ev = " + ".join(evidences) if evidences else "Combined Prosody & Acoustic Timbre"
            
            pairs_list.append({
                'Pair': f"{spk_a} ↔ {spk_b}",
                'Combined Distance': round(c_dist, 3),
                'Embedding Distance': round(e_dist, 3),
                'MFCC Distance': round(m_dist, 3),
                'Main Evidence': main_ev
            })
            
    df_pairs = pd.DataFrame(pairs_list).sort_values('Combined Distance')
    table2_path = os.path.join(OUTPUT_DIR, 'table2_pairwise_distances.csv')
    df_pairs.to_csv(table2_path, index=False)
    print(f"   Table 2 saved to: {table2_path}")
    
    print("\n   Top 5 Closest Speaker Pairs:")
    print(df_pairs.head(5)[['Pair', 'Combined Distance', 'Embedding Distance', 'Main Evidence']].to_string(index=False))
    print("\n   Top 5 Most Distant Speaker Pairs:")
    print(df_pairs.tail(5)[['Pair', 'Combined Distance', 'Embedding Distance', 'Main Evidence']].to_string(index=False))
    
    total_pipeline_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ ACOUSTIC ANALYSIS COMPLETED SUCCESSFULLY IN {total_pipeline_time:.1f} SECONDS ({total_pipeline_time/60:.2f} MIN)")
    print("=" * 80)

if __name__ == '__main__':
    main()
