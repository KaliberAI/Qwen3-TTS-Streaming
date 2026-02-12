import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import time
import numpy as np

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()

def run_streaming_test(
    model,
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    label: str = "streaming",
):
    """Run streaming generation and return timing stats."""
    start = time.time()
    chunks = []
    chunk_sizes = []
    first_chunk_time = None
    chunk_count = 0
    sample_rate = 24000

    for chunk, chunk_sr in model.stream_generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
        overlap_samples=0,
    ):
        chunk_count += 1
        chunks.append(chunk)
        chunk_sizes.append(len(chunk))
        sample_rate = chunk_sr
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    total_time = time.time() - start
    final_audio = np.concatenate(chunks) if chunks else np.array([])

    # Calculate audio duration and chunk stats
    audio_duration = len(final_audio) / sample_rate if sample_rate > 0 else 0
    avg_chunk_samples = np.mean(chunk_sizes) if chunk_sizes else 0
    avg_chunk_duration = avg_chunk_samples / sample_rate if sample_rate > 0 else 0

    return {
        "label": label,
        "first_chunk_time": first_chunk_time,
        "total_time": total_time,
        "chunk_count": chunk_count,
        "audio": final_audio,
        "sample_rate": sample_rate,
        "audio_duration": audio_duration,
        "avg_chunk_samples": avg_chunk_samples,
        "avg_chunk_duration": avg_chunk_duration,
    }

def main():
    total_start = time.time()

    # Streaming parameters
    EMIT_EVERY = 4
    DECODE_WINDOW = 80

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)
    
    start = time.time()
    # Load the model
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    log_time(start, "Model loaded")

    # Test parameters
    test_text = "I understand that you're feeling anxious right now. Trust me, you're in a safe place and we're here to help you. Take some deep, slow breaths. Everything is going to be alright."
    speaker = "aiden"
    language = "English"
    instruct = "Use a calming, steadying tone to lower to patient's anxiety and provide a sense of safety."

    results = []

    # ============== Test 1: Standard generation ==============
    print("\n" + "=" * 60)
    print("Test 1: Standard (non-streaming) generation")
    print("=" * 60)

    start = time.time()
    wavs, sr = model.generate_custom_voice(
        text=test_text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )
    standard_time = time.time() - start
    standard_audio_duration = len(wavs[0]) / sr
    standard_rtf = standard_time / standard_audio_duration
    print(f"[{standard_time:.2f}s] Standard generation complete")
    print(f"Audio duration: {standard_audio_duration:.2f}s, RTF: {standard_rtf:.2f}")
    sf.write("output_instruction_standard.wav", wavs[0], sr)
    results.append({
        "label": "standard",
        "total_time": standard_time,
        "audio_duration": standard_audio_duration,
    })

    # ============== Test 2: Streaming WITHOUT optimizations ==============
    print("\n" + "=" * 60)
    print("Test 2: Streaming WITHOUT optimizations")
    print("=" * 60)

    result_baseline = run_streaming_test(
        model, test_text, language, speaker, instruct,
        emit_every_frames=EMIT_EVERY,
        decode_window_frames=DECODE_WINDOW,
        label="streaming_baseline_instruct",
    )
    results.append(result_baseline)
    sf.write("output_instruction_streaming_baseline.wav", result_baseline["audio"], result_baseline["sample_rate"])
    rtf = result_baseline['total_time'] / result_baseline['audio_duration'] if result_baseline['audio_duration'] > 0 else 0
    print(f"First chunk: {result_baseline['first_chunk_time']:.2f}s, Total: {result_baseline['total_time']:.2f}s, Chunks: {result_baseline['chunk_count']}")
    print(f"Audio duration: {result_baseline['audio_duration']:.2f}s, Chunk duration: {result_baseline['avg_chunk_duration']*1000:.0f}ms, RTF: {rtf:.2f}")


    # ============== Test 3: Streaming WITH optimizations ==============
    print("\n" + "=" * 60)
    print("Test 3: Streaming WITH torch.compile (decoder + talker + codebook)")
    print("=" * 60)

    print("\nEnabling streaming optimizations...")
    model.enable_streaming_optimizations(
        decode_window_frames=DECODE_WINDOW,
        use_compile=True,
        use_cuda_graphs=False,
        compile_mode="reduce-overhead",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    # Warmup runs
    warmup_texts = [
        "This is a short test.",
        "Hello, how are you? This is a second warmup run.",
    ]

    print("\nWarmup runs (compilation happens here)...")
    for i, warmup_text in enumerate(warmup_texts, 1):
        warmup_result = run_streaming_test(
            model, warmup_text, language, speaker, instruct,
            emit_every_frames=EMIT_EVERY,
            decode_window_frames=DECODE_WINDOW,
            label=f"warmup_{i}",
        )
        warmup_rtf = warmup_result['total_time'] / warmup_result['audio_duration'] if warmup_result['audio_duration'] > 0 else 0
        print(f"  Warmup {i}: {warmup_result['total_time']:.2f}s, Audio: {warmup_result['audio_duration']:.2f}s, RTF: {warmup_rtf:.2f}")

    # Optimized test run
    print("\nOptimized test run...")
    result_optimized = run_streaming_test(
        model, test_text, language, speaker, instruct,
        emit_every_frames=EMIT_EVERY,
        decode_window_frames=DECODE_WINDOW,
        label="streaming_optimized_instruct",
    )
    results.append(result_optimized)
    sf.write("output_instruction_streaming_optimized.wav", result_optimized["audio"], result_optimized["sample_rate"])
    rtf = result_optimized['total_time'] / result_optimized['audio_duration'] if result_optimized['audio_duration'] > 0 else 0
    print(f"First chunk: {result_optimized['first_chunk_time']:.2f}s, Total: {result_optimized['total_time']:.2f}s, Chunks: {result_optimized['chunk_count']}")
    print(f"Audio duration: {result_optimized['audio_duration']:.2f}s, Chunk duration: {result_optimized['avg_chunk_duration']*1000:.0f}ms, RTF: {rtf:.2f}")

    # ============== Summary ==============
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_result = results[1]  # streaming_baseline_instruct
    baseline_audio_dur = baseline_result.get("audio_duration", 0)
    baseline_rtf = baseline_result["total_time"] / baseline_audio_dur if baseline_audio_dur > 0 else 0

    print(f"\n{'Method':<35} {'1st Chunk':>10} {'Total':>8} {'Audio':>8} {'RTF':>6} {'Chunks':>7} {'RTF Speedup':>12}")
    print("-" * 100)

    # Standard generation
    std = results[0]
    std_rtf = std['total_time'] / std['audio_duration'] if std.get('audio_duration', 0) > 0 else 0
    print(f"{'Standard (no streaming)':<35} {'N/A':>10} {std['total_time']:>7.2f}s {std.get('audio_duration', 0):>7.2f}s {std_rtf:>6.2f} {'N/A':>7} {'N/A':>12}")

    for r in results[1:]:
        # Skip warmup results in summary table
        if r['label'].startswith("warmup_"):
            continue

        first = r.get("first_chunk_time", 0)
        total = r["total_time"]
        audio_dur = r.get("audio_duration", 0)
        rtf = total / audio_dur if audio_dur > 0 else 0
        chunks = r.get("chunk_count", 0)
        speedup_rtf = baseline_rtf / rtf if rtf > 0 else 0
        print(f"{r['label']:<35} {first:>9.2f}s {total:>7.2f}s {audio_dur:>7.2f}s {rtf:>6.2f} {chunks:>7} {speedup_rtf:>11.2f}x")

    print(f"\n[{time.time() - total_start:.2f}s] TOTAL SCRIPT TIME")


if __name__ == "__main__":
    main()

