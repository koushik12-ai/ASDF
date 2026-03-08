# Voice AI Pipeline: Architecture & Design Report

## Overview

This document describes the pipeline, architecture, and design of the **Voice AI Assistant** project — a fully functional, real-time conversational Voice AI system integrating audio capture, noise suppression, voice activity detection (VAD), automatic speech recognition (ASR), an LLM reasoning layer with Retrieval-Augmented Generation (RAG), and text-to-speech (TTS) output.

---

## Core Architecture and Modules

The pipeline is modularized into six sequential stages, orchestrated by `main.py`:

### Module 1 — Audio Input (`pipeline/audio_input.py`)
- **Class**: `AudioStreamingClient`
- Captures real-time audio at **16kHz** using `sounddevice`.
- Outputs raw audio as a continuous byte stream to the next stage.

### Module 2 — Noise Suppressor (`pipeline/noise_suppressor.py`)
- **Class**: `NoiseSuppressor`
- Applies a NumPy-based **amplitude noise gate**.
- Audio chunks below a configurable dB threshold (default −40 dB) are silenced; louder chunks pass through unchanged.

### Module 3 — Voice Activity Detection (`pipeline/vad.py`)
- **Class**: `VoiceActivityDetector`
- Uses **RMS energy detection** mapped to an aggressiveness scale (0–3).
- Maintains a ring buffer and state machine to cleanly capture complete spoken sentences, avoiding word cutoffs.

### Module 4 — Automatic Speech Recognition (`pipeline/asr.py`)
- **Class**: `SpeechToTextEngine`
- Converts isolated speech segments into text.
- Currently a stub — integrate **Whisper** (local) or **Groq Whisper** (cloud) for production use.

### Module 5 — Intelligence & RAG Engine (`pipeline/intelligence_engine.py`)
- **Class**: `IntelligenceEngine`
- Acts as the "brain" of the assistant.
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free).
- **Vector Store**: FAISS for fast similarity search over the knowledge base.
- **LLM**: Groq Cloud API (`llama-3.1-8b-instant` or `llama-3.3-70b-versatile`) via OpenAI-compatible endpoint.
- **RAG Flow**: Embeds query → retrieves relevant context from FAISS → injects context into LLM prompt → returns answer.

### Module 6 — Text-to-Speech (`pipeline/tts_engine.py`)
- **Class**: `TTSEngine`
- Converts AI text response to natural speech using `edge_tts`.
- Generates an `.mp3` file asynchronously and plays it via the OS default media player.

---

## Data Workflow

```
Microphone
    │
    ▼
[Module 1] AudioStreamingClient   — raw audio bytes @ 16kHz
    │
    ▼
[Module 2] NoiseSuppressor        — noise-gated audio bytes
    │
    ▼
[Module 3] VoiceActivityDetector  — complete speech segment
    │
    ▼
[Module 4] SpeechToTextEngine     — transcribed text string
    │
    ▼
[Module 5] IntelligenceEngine     — FAISS retrieval + Groq LLM response
    │
    ▼
[Module 6] TTSEngine              — spoken MP3 audio output
    │
    ▼
Speaker
```

---

## Alternative Interfaces

| Interface | Entry Point | Description |
|---|---|---|
| **Modular Pipeline** | `main.py` | Full 6-module real-time pipeline |
| **Streamlit Web UI** | `app/streamlit_app.py` | Browser-based chat + voice recorder |
| **Standalone Bot** | `scripts/voice_bot_standalone.py` | Self-contained script with pyttsx3 TTS |
| **RAG Engine Only** | `rag/rag_engine.py` | LangChain FAISS RAG without voice I/O |

---

## Design Advantages

- **Low latency**: Heavy FAISS vector operations run **locally** without network calls.
- **Cloud LLM**: Groq API delivers ultra-fast LLM inference (sub-second tokens/s) via its LPU hardware.
- **Modular**: Each stage can be swapped independently (e.g., replace `edge_tts` with ElevenLabs, or Groq with OpenAI).
- **Python 3.14 compatible**: Uses `sounddevice` (replaces PyAudio) and energy-based VAD (replaces WebRTC VAD).
