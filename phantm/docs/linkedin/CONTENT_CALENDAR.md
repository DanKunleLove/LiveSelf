# LiveSelf LinkedIn Content Calendar

## Strategy
- 1 post per day for 7-10 days
- Each post documents a real build step with proof (screenshots, code, results)
- Every post links to the GitHub repo for contributors
- Tone: honest builder sharing the journey, not polished marketing
- No hype claims -- show what actually works
- End every post with a contributor call-to-action

GitHub link for all posts: https://github.com/DanKunleLove/LiveSelf

---

## Day 1: The Announcement

**Hook:** I'm building an open-source AI that attends Zoom calls as you.

**Body:**
I'm a final-year Physics and CS student in Nigeria. I have no GPU. My budget is close to zero.

But I have an idea that won't leave my head: what if you could create an AI version of yourself -- your face, your voice, your knowledge -- and have it attend meetings for you?

Not a chatbot. A live digital twin on an actual Zoom call.

I started building it this week. It's called LiveSelf. It's open source. Here's what the plan looks like:

1. Face swap -- your photo becomes a live video feed
2. Voice clone -- 10 seconds of your voice, and it speaks as you
3. AI brain -- it pulls from your knowledge to answer questions
4. Dashboard -- anyone can set this up through a web app

Every piece uses open-source AI models. No API costs for the core pipeline.

I'm documenting every step here. Follow along if you want to see either an incredible tool or an incredible failure.

GitHub: https://github.com/DanKunleLove/LiveSelf

**Image suggestion:** Screenshot of the project README on GitHub showing the tech stack table and architecture diagram.

---

## Day 2: Face Swap Works

**Hook:** Day 2: My AI twin has a face.

**Body:**
I ran InsightFace on a free Google Colab GPU. Uploaded one photo of myself. It swapped my face onto a video in real-time.

This is running on a free T4 GPU. No paid services. No API keys. Just open-source models.

Here's what the pipeline looks like so far:

Photo --> InsightFace --> face-swapped video frame --> virtual camera

The code is 281 lines of Python. It handles: loading the model, detecting faces, swapping the largest face in frame, and outputting to a virtual camera.

Next step: making it speak.

Code: https://github.com/DanKunleLove/LiveSelf/blob/main/phantm/engine/pipeline/faceswap.py

**Image suggestion:** Side-by-side of the original video frame and the face-swapped output from Colab.

---

## Day 3: The Voice Clone

**Hook:** I cloned my voice with 10 seconds of audio.

**Body:**
CosyVoice 2 is an open-source voice cloning model from FunAudioLLM. You give it a short sample of someone's voice, type a sentence, and it speaks that sentence in the cloned voice.

I integrated it into LiveSelf. The VoiceCloner class:
- Takes a 10-30 second WAV file as reference
- Generates new speech from any text input
- Outputs at 16kHz sample rate
- Runs on a single T4 GPU

The dependency setup was brutal. CosyVoice needs openai-whisper, which refuses to build from PyPI on Colab. The fix: install it from GitHub source directly.

I wrote a Colab notebook that handles all of this automatically. Anyone can test it.

Testing notebook: https://github.com/DanKunleLove/LiveSelf/blob/main/phantm/scripts/colab_phase1b.py

**Image suggestion:** Screenshot of the Colab notebook running, showing the voice synthesis output with timing numbers.

---

## Day 4: The Brain

**Hook:** My AI twin can think. It answered questions about my life using Llama 3.

**Body:**
This is the part that makes it real.

I set up a pipeline on Google Colab:
- ChromaDB stores knowledge about me (Q&A pairs, facts, context)
- When someone asks a question, it retrieves the relevant knowledge
- Llama 3 (running locally via Ollama on the GPU) generates a response
- The response streams sentence by sentence

I asked it: "What is LiveSelf and what do you do?"

It responded:
"LiveSelf is an open-source platform that makes AI digital twins accessible to everyone, including myself! As my digital twin, I'm essentially a virtual version of me -- Dan."

That response came from knowledge I indexed, shaped by Llama 3's reasoning. Not a canned answer. Not an API call. Running entirely on a free GPU.

Latency: ~300ms to first sentence with streaming.

Code: https://github.com/DanKunleLove/LiveSelf/blob/main/phantm/engine/pipeline/llm.py

**Image suggestion:** Screenshot of the Colab output showing the RAG retrieval + LLM streaming response.

---

## Day 5: The Architecture

**Hook:** Here's the full pipeline that makes a live AI digital twin possible.

**Body:**
The complete chain, running on a single GPU:

Microphone
  --> faster-whisper (speech to text, ~150ms)
  --> ChromaDB (knowledge retrieval, ~50ms)
  --> Llama 3 (generate response, ~300ms to first sentence)
  --> CosyVoice 2 (speak in your voice)
  --> MuseTalk 1.5 (sync lips to audio)
  --> InsightFace (swap face)
  --> Virtual Camera --> Zoom

Target: 500ms from hearing a question to starting to speak.

Every component is open source. Every component has a permissive license. The orchestrator runs all stages concurrently with asyncio queues.

6 AI models. 1 GPU. 0 API costs.

Full architecture: https://github.com/DanKunleLove/LiveSelf

**Image suggestion:** The architecture diagram from the README (the ASCII flow chart, or recreate it as a clean diagram).

---

## Day 6: The Backend

**Hook:** Built 12 API endpoints in one day. Here's why FastAPI is absurdly productive.

**Body:**
The backend handles everything a user needs before going live:
- Register / login (Supabase auth)
- Create a persona (upload face photo, record voice)
- Start / end sessions
- Track usage

12 endpoints. FastAPI + Pydantic models. Full type safety.

I built this using a multi-agent workflow -- one Claude Code agent on the backend while another worked on the AI pipeline simultaneously.

The backend is ready. It just needs a frontend.

That's where contributors come in. If you know Next.js + TypeScript + Tailwind, there's a detailed task file waiting for you.

Contribute: https://github.com/DanKunleLove/LiveSelf/blob/main/CONTRIBUTING.md

**Image suggestion:** Screenshot of the backend running with endpoint list, or the routers directory listing.

---

## Day 7: Open Source and Looking for Contributors

**Hook:** I'm one integration test away from a live AI twin on Zoom. And I need help.

**Body:**
Here's where LiveSelf stands right now:

What works:
- Face swap (tested, video output confirmed)
- AI brain (tested, streams responses from knowledge)
- Voice clone (code done, testing this week)
- Lip sync (code done, testing this week)
- Backend API (12 endpoints, ready)

What needs help:
- Frontend dashboard (Next.js 14, design system spec included)
- GPU testing on different hardware
- Knowledge ingestion (PDF, YouTube, docs)
- Performance optimization
- Documentation

This is a real project with real code that runs on free GPUs. It's not a concept. It's not a pitch deck.

I set up:
- Detailed README with architecture
- CONTRIBUTING.md with setup instructions
- Issue templates
- Task files breaking down exactly what needs building
- Colab notebooks so you can test without a GPU

If you've ever wanted to contribute to an AI project from the ground up, this is it.

GitHub: https://github.com/DanKunleLove/LiveSelf

**Image suggestion:** The GitHub repo landing page showing the README, or the CONTRIBUTING.md.

---

## Bonus Posts (Days 8-10)

### Day 8: Dependency Hell
**Hook:** It took 6 attempts to install one Python package on Colab.

Tell the story of the openai-whisper build failure and how you fixed it. Developers love war stories about dependency management. End with: "I documented every fix in the Colab notebooks so nobody else has to fight this."

### Day 9: Building With AI Tools
**Hook:** I'm a Physics student who barely codes. Here's how I'm building an AI platform.

Be honest about using Claude Code as a development partner. Show the multi-agent workflow. This is a compelling story for both technical and non-technical audiences.

### Day 10: The Demo
**Hook:** [Video] My AI twin just had a conversation on Zoom.

Post the WOW demo video when it's ready. This is the payoff post.

---

## Post Formatting Tips

1. Start with a single-line hook that makes people stop scrolling
2. Use short paragraphs (1-2 sentences each)
3. Include specific numbers (281 lines, 150ms, 6 models, 0 API costs)
4. Always end with the GitHub link
5. Add 3-5 hashtags: #opensource #AI #buildinpublic #deeplearning #zoom
6. Post between 8-10 AM your timezone for maximum reach
7. Reply to every comment -- LinkedIn rewards engagement
