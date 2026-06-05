#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys


def find_participant_dirs(data_dir):
    """Return a list of participant UUID directories that contain .wav files."""
    dirs = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        if any(f.endswith(".wav") for f in os.listdir(full)):
            dirs.append(full)
    return dirs


def find_wav_files(participant_dir, filter_pattern=None):
    regex = re.compile(filter_pattern) if filter_pattern else None
    wav_files = []
    for f in sorted(os.listdir(participant_dir)):
        if not f.endswith(".wav"):
            continue
        if regex and not regex.search(f):
            continue
        wav_files.append(os.path.join(participant_dir, f))
    return wav_files


def transcribe_faster_whisper(wav_files, model):
    results = []
    for wav_path in wav_files:
        print(f"  transcribing: {os.path.basename(wav_path)}")
        segments, _ = model.transcribe(wav_path, language="en")
        text = " ".join(segment.text.strip() for segment in segments)
        results.append((os.path.basename(wav_path), text))
    return results


def transcribe_qwen_asr(wav_files, model):
    print(f"  batching {len(wav_files)} file(s)")
    out = model.transcribe(audio=wav_files)
    results = []
    for wav_path, result in zip(wav_files, out):
        results.append((os.path.basename(wav_path), result.text.strip()))
    return results


def write_csv(participant_dir, rows):
    csv_path = os.path.join(participant_dir, "transcriptions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text"])
        writer.writerows(rows)
    print(f"  wrote: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe .wav files from the data directory."
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "/data"),
        help="Path to data directory (default: /data or $DATA_DIR)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Regex filter applied to filenames (e.g. 'trial' or '00[1-3]')",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name (default depends on backend)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe even if transcriptions.csv already exists",
    )
    parser.add_argument(
        "--backend",
        choices=["faster-whisper", "qwen-asr"],
        default=None,
        help="Force a specific backend (auto-detected if omitted)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    participant_dirs = find_participant_dirs(args.data_dir)
    if not participant_dirs:
        print("No participant directories with .wav files found.")
        sys.exit(0)

    # Auto-detect backend
    backend = args.backend
    if backend is None:
        try:
            import faster_whisper  # noqa: F401
            backend = "faster-whisper"
        except ImportError:
            pass
    if backend is None:
        try:
            import qwen_asr  # noqa: F401
            backend = "qwen-asr"
        except ImportError:
            pass
    if backend is None:
        print("Error: no backend available. Install faster-whisper or qwen-asr.", file=sys.stderr)
        sys.exit(1)

    # Load model once
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel
        model_name = args.model or "large-v3-turbo"
        print(f"Loading faster-whisper model: {model_name} (int8, cpu)")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        transcribe_fn = transcribe_faster_whisper
    else:
        import torch
        from qwen_asr import Qwen3ASRModel
        model_name = args.model or "Qwen/Qwen3-ASR-1.7B"
        print(f"Loading Qwen3-ASR model: {model_name} (bfloat16, cuda)")
        model = Qwen3ASRModel.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda:0"
        )
        transcribe_fn = transcribe_qwen_asr

    print(f"Backend: {backend}")
    print(f"Processing {len(participant_dirs)} participant(s)\n")

    for pdir in participant_dirs:
        participant_id = os.path.basename(pdir)
        csv_path = os.path.join(pdir, "transcriptions.csv")

        if os.path.exists(csv_path) and not args.overwrite:
            print(f"[{participant_id}] skip (transcriptions.csv exists)")
            continue

        wav_files = find_wav_files(pdir, args.filter)
        if not wav_files:
            print(f"[{participant_id}] no matching .wav files")
            continue

        print(f"[{participant_id}] {len(wav_files)} file(s)")
        rows = transcribe_fn(wav_files, model)
        write_csv(pdir, rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
