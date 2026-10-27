# CPSC 254 - Assignment 4

This project uses Python for news classification, an LLM agent, syllabus question answering (RAG), speech recognition, and text-to-speech.

## Setup

Use Python 3.13. Open a terminal in the `topic-classifier` folder and install all Python packages:

```powershell
python -m pip install -r requirements.txt
```

For speech recognition, install FFmpeg separately (it is not a Python package):

```powershell
winget install --id Gyan.FFmpeg -e
```

Create a `.env` file in `topic-classifier` with your Nautilus settings:

```dotenv
OPENAI_API_KEY=your_nautilus_token
NAUTILUS_BASE_URL=https://ellm.nrp-nautilus.io/v1
NAUTILUS_MODEL=gpt-oss
```

Confirm the endpoint and model name with your course instructions. Put `CPSC 254_SYL.pdf` in the same folder.

## Part 1: Text Classification

Train and compare four MLP models on AG News:

```powershell
python topic_classifier.py
```

## Part 2: LLM Classification Agent

Classify one article:

```powershell
python classification_agent.py --article "Apple released a new smartphone."
```

Compare both classifiers on 200 test articles (requires Part 1 saved models):

```powershell
python classification_agent.py --evaluate --sample-size 200
```

## Part 3: RAG Application

Build the syllabus database once:

```powershell
python rag.py --build --pdf "CPSC 254_SYL.pdf"
```

Ask a question and run the evaluation:

```powershell
python rag.py --question "What is the grading policy?"
python rag_eval.py
```

## Part 4: Speech

Transcribe a recorded question and get a RAG answer:

```powershell
python stt_rag.py --audio test-recorded-audio.mp3
```

Generate and play a spoken answer:

```powershell
python tts_rag.py --question "Who is the instructor?"
```

Save the spoken answer:

```powershell
python tts_rag.py --question "Who is the instructor?" --save-audio outputs/answer.wav
```

## Notes

The AG News dataset downloads automatically. Parts 2-4 need Nautilus access, and Parts 3-4 need the saved Chroma syllabus index. Results are saved in `outputs/`. Never share `.env` or API tokens.
