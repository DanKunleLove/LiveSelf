# LiveSelf

**Attend Zoom, WhatsApp, and Google Meet calls as your AI twin. Open source. Free.**

## What is LiveSelf?

LiveSelf lets you create a live AI version of yourself -- your face, your voice, your knowledge -- that can attend real-time video calls for you. Upload a photo, record 10 seconds of your voice, add your knowledge base, and your AI twin handles the rest.

## How It Works

1. **Upload your face** -- One clear photo
2. **Record your voice** -- 10 seconds is all it takes
3. **Add your knowledge** -- PDFs, notes, Q&A pairs
4. **Go Live** -- Your avatar appears on Zoom as you

## Tech Stack

| Layer | Tool | License |
|-------|------|---------|
| Face Swap | [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) | MIT |
| Lip Sync | [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) | MIT |
| Voice Clone | [CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice) | Apache 2.0 |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT |
| AI Brain | [Ollama](https://github.com/ollama/ollama) + Llama 3 | MIT |
| Knowledge Base | [ChromaDB](https://github.com/chroma-core/chroma) | Apache 2.0 |

## Quick Start

Requires Python 3.11 and a GPU (NVIDIA, 8GB+ VRAM)

```bash
git clone https://github.com/DanKunleLove/LiveSelf.git
cd LiveSelf
bash scripts/setup_dev.sh
```

## Status

Under active development -- Phase 1 (AI Pipeline)

See [ROADMAP.md](docs/ROADMAP.md) for the full build plan.

## Contributing

PRs welcome. See the docs/agents/ folder for current development tasks.

## License

MIT -- free to use, modify, and commercialize.
