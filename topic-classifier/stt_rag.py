import argparse
import json
from pathlib import Path

from rag import ask, load_index


def record_microphone(seconds: float, sample_rate: int):
    import numpy as np
    import sounddevice as sd
    if seconds <= 0:
        raise ValueError('Recording duration must be positive')
    print(f'Recording microphone for {seconds:g} seconds...', flush=True)
    recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                       channels=1, dtype='float32')
    sd.wait()
    audio = np.asarray(recording, dtype=np.float32).reshape(-1)
    if not np.any(np.abs(audio) > 1e-6):
        raise ValueError('No audible microphone signal was recorded')
    print('Recording complete.', flush=True)
    return audio


def transcribe(audio_path: Path | None, microphone: bool, duration: float,
               whisper_model: str) -> str:
    import whisper
    model = whisper.load_model(whisper_model)
    if microphone:
        # Whisper accepts a float32 mono waveform at 16 kHz.
        audio = record_microphone(duration, 16000)
    else:
        if audio_path is None or not audio_path.is_file():
            raise FileNotFoundError(f'Audio file not found: {audio_path}')
        # Whisper loads file formats via FFmpeg (must be installed on PATH).
        audio = str(audio_path)
    result = model.transcribe(audio, language='en', fp16=False)
    text = str(result.get('text', '')).strip()
    if not text:
        raise ValueError('Whisper did not recognize any speech')
    return text


def main():
    parser = argparse.ArgumentParser(description='Speech-to-text syllabus RAG')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--audio', type=Path, help='Existing spoken-question audio file')
    source.add_argument('--microphone', action='store_true', help='Record microphone audio')
    parser.add_argument('--duration', type=float, default=6, help='Microphone recording seconds')
    parser.add_argument('--whisper-model', default='base', help='Whisper model, e.g. tiny/base/small')
    parser.add_argument('--index-dir', type=Path, default=Path('chroma_syllabus'))
    parser.add_argument('--output', type=Path, default=Path('outputs/stt_rag_result.json'))
    args = parser.parse_args()

    question = transcribe(args.audio, args.microphone, args.duration, args.whisper_model)
    print(f'\nTRANSCRIPTION: {question}', flush=True)
    response = ask(question, load_index(args.index_dir))
    print(f'\nRAG ANSWER: {response["answer"]}', flush=True)
    for n, source_doc in enumerate(response['sources'], 1):
        print(f'SOURCE S{n}: syllabus page {source_doc["page"]}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'transcription': question, **response},
                                      indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved evidence: {args.output}', flush=True)


if __name__ == '__main__':
    main()
