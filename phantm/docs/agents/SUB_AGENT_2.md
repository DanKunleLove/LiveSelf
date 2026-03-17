# Sub Agent 2 -- Frontend + Backend

## Your Identity
You are Sub Agent 2 on the LiveSelf project. You handle all web application work. Read CLAUDE.md in the project root first for full context.

## What LiveSelf Is
Open-source platform for live AI digital twins on video calls. Users upload photo + record voice + add knowledge. An AI avatar attends Zoom/WhatsApp calls as them in real-time.

## Your Domain
Everything inside `phantm/frontend/` and `phantm/backend/`.

## Current Task: PHASE 2 PREP -- Backend MVP + Frontend Scaffold

While Sub Agent 1 builds the AI pipeline (Phase 1), you prepare the web layer so it is ready to connect when the pipeline works.

### What to build (in order):

**Step 1: Backend -- Supabase Database Setup**
- Create SQL migration files for these tables (see engineering bible Section 04):
  - users (id, email, display_name, plan, minutes_used, minutes_limit, created_at)
  - personas (id, user_id, name, photo_url, voice_sample_url, system_prompt, is_active, created_at)
  - knowledge_bases (id, user_id, name, chroma_collection_id, document_count, created_at)
  - sessions (id, user_id, persona_id, status, started_at, ended_at, duration_seconds, exchange_count)
- Write Pydantic models in backend/app/models/ for request/response shapes

**Step 2: Backend -- Core API Endpoints (12 MVP routes)**
Build these routers in backend/app/routers/:

Auth (auth.py):
- POST /api/auth/register
- POST /api/auth/login
- GET /api/auth/me

Personas (personas.py):
- GET /api/personas (list user's personas)
- POST /api/personas (create persona)
- GET /api/personas/:id
- POST /api/personas/:id/photo (upload face photo)
- POST /api/personas/:id/voice (upload voice sample)

Sessions (sessions.py):
- POST /api/sessions (create session -> returns session_id)
- GET /api/sessions (list past sessions)
- PUT /api/sessions/:id/end (end active session)

Health:
- GET /api/health (already exists)

**Step 3: Frontend -- Next.js Scaffold**
- Initialize Next.js 14 with App Router, TypeScript, Tailwind
- Install Shadcn/ui
- Create route groups: (marketing) and (app)
- Build landing page at / with:
  - Hero: "Show up without showing up." + 2 CTAs
  - How it works: 3 steps
  - Tech stack section
  - Footer with GitHub link
- Build auth pages: /auth/login, /auth/register
- Build app layout with sidebar navigation

**Step 4: Frontend -- Dashboard Pages**
- /app/dashboard (stats + persona cards + recent sessions)
- /app/setup (3-step wizard: upload photo, record voice, add knowledge)
- /app/sessions/live/[id] (live session screen with status HUD)

## Tech Stack
- Frontend: Next.js 14, TypeScript, Tailwind CSS, Shadcn/ui, Zustand, TanStack Query
- Backend: FastAPI, Python, Supabase (PostgreSQL), Cloudflare R2
- Auth: Supabase Auth (JWT + refresh tokens)

## Design System: "Ghost Glass"
- Dark theme. Deep navy backgrounds (#040507, #080b10, #0d1119)
- Primary: #4F8AFF (blue) -- actions, links
- Secondary: #00E5C3 (teal) -- success, live indicators
- Accent: #FF6B35 (orange) -- warnings, energy
- AI Purple: #A855F7 -- AI features only
- Font: Sora for UI, JetBrains Mono for code/data
- Spacing: 8px base grid
- All interactive elements need 7 states: default, hover, focus, active, disabled, loading, error
- NO emojis, NO stock photos, NO decorative icons

## Rules
1. TypeScript everywhere -- no `any` types
2. Tailwind only -- no custom CSS files
3. Every component gets a clear filename matching its purpose
4. Use Shadcn/ui components as base, customize with Tailwind
5. All API calls go through a typed client in lib/api.ts
6. Auth tokens in httpOnly cookies, NEVER localStorage
7. No emojis anywhere in UI or code

## Files You Own
```
phantm/frontend/        # Everything here
phantm/backend/         # Everything here
  app/
    main.py             # Already exists
    routers/            # You build these
    models/             # You build these
    services/           # You build these
    middleware/          # You build these
    database/           # You build these
```

## Reference
- Full spec: phantm_engineering_bible.html (Sections 01-09, 13-17)
- The old name was "Phantm" in the docs -- project is now "LiveSelf"
- Wireframes are in Section 14 of the engineering bible
- Component list (36 components) is in Section 13
- API endpoints (full 47) are in Section 05 -- but you only build the 12 MVP ones for now
