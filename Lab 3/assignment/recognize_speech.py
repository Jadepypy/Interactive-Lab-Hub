import queue
import sys
import time
import sounddevice as sd
import numpy as np
import json
from vosk import Model, KaldiRecognizer


def record_and_recognize(
    model_lang="en-us",
    device=None,
    samplerate=None,
    filename=None,
    blocksize=8000,
    silence_threshold=500,   # Adjust depending on mic sensitivity
    silence_duration=1.5,    # seconds of silence before auto-stop
    max_duration=10.0         # hard limit (optional)
):
    """
    Record user's speech and stop automatically after a few seconds of silence.
    Returns a list of recognized phrases (strings).
    """

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Initialize device and sample rate
    if samplerate is None:
        device_info = sd.query_devices(device, "input")
        samplerate = int(device_info["default_samplerate"])

    # Load Vosk model
    model = Model(lang=model_lang)
    rec = KaldiRecognizer(model, samplerate)

    # Optional file to dump raw audio
    dump_fn = open(filename, "wb") if filename else None

    results = []
    silence_start = None
    start_time = time.time()

    print("#" * 80)
    print("🎙️  Listening... Speak now.")
    print("Auto-stops after a pause.")
    print("#" * 80)

    try:
        with sd.RawInputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                data = q.get()
                if dump_fn:
                    dump_fn.write(data)

                # --- detect silence using RMS energy ---
                audio_block = np.frombuffer(data, dtype=np.int16)
                volume_norm = np.linalg.norm(audio_block) / len(audio_block)
                is_silent = volume_norm < silence_threshold

                # feed to recognizer
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result()).get("text", "").strip()
                    if result:
                        results.append(result)
                        print(f"✅ Recognized: {result}")
                else:
                    partial = json.loads(rec.PartialResult()).get("partial", "")
                    if partial:
                        print(f"…Partial: {partial}", end="\r")

                # silence tracking
                if is_silent:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        print("\n🤫 Silence detected. Stopping.")
                        break
                else:
                    silence_start = None  # reset if speech resumes

                if time.time() - start_time > max_duration:
                    print("\n⏰ Max duration reached. Stopping.")
                    break

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        if dump_fn:
            dump_fn.close()

    return results