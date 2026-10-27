import argparse
import json
import platform
from pathlib import Path

from rag import ask, load_index


def speak_answer(text: str, save_audio: Path | None, play: bool, rate: int):
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    if save_audio:
        save_audio.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, str(save_audio.resolve()))
        engine.runAndWait()
        if not save_audio.is_file() or save_audio.stat().st_size == 0:
            raise RuntimeError('TTS engine did not create a nonempty audio file')
        print(f'SAVED AUDIO: {save_audio}', flush=True)
        # On Windows, play the saved WAV to avoid competing pyttsx3 operations.
        if play and platform.system() == 'Windows' and save_audio.suffix.lower() == '.wav':
            import winsound
            winsound.PlaySound(str(save_audio.resolve()), winsound.SND_FILENAME)
            print('PLAYBACK: completed', flush=True)
            return
    if play:
        engine.say(text)
        engine.runAndWait()
        print('PLAYBACK: completed', flush=True)
    else:
        print('PLAYBACK: skipped (--no-play)', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Text-to-speech syllabus RAG')
    parser.add_argument('--question', required=True, help='Question to ask the syllabus RAG')
    parser.add_argument('--index-dir', type=Path, default=Path('chroma_syllabus'))
    parser.add_argument('--save-audio', type=Path, help='Save synthesized speech as a WAV file')
    parser.add_argument('--no-play', action='store_true', help='Save audio without speaker playback')
    parser.add_argument('--rate', type=int, default=150, help='Speech rate in words per minute')
    parser.add_argument('--output', type=Path, default=Path('outputs/tts_rag_result.json'))
    args = parser.parse_args()
    if args.no_play and not args.save_audio:
        parser.error('--no-play requires --save-audio to demonstrate speech synthesis')
    if args.save_audio and args.save_audio.suffix.lower() != '.wav':
        parser.error('--save-audio must use a .wav filename')

    print(f'QUESTION: {args.question}', flush=True)
    result = ask(args.question, load_index(args.index_dir))
    print(f'RAG ANSWER: {result["answer"]}', flush=True)
    for n, source_doc in enumerate(result['sources'], 1):
        print(f'SOURCE S{n}: syllabus page {source_doc["page"]}', flush=True)
    speak_answer(result['answer'], args.save_audio, not args.no_play, args.rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**result, 'audio_file': str(args.save_audio) if args.save_audio else None,
                                       'playback_requested': not args.no_play},
                                      indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved evidence: {args.output}', flush=True)


if __name__ == '__main__':
    main()
