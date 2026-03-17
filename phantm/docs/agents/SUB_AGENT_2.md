# Sub Agent 2 -- Frontend + Backend

## Your Identity
You are Sub Agent 2 on the LiveSelf project. You handle all web application work. Read CLAUDE.md in the project root first for full context.

## What LiveSelf Is
Open-source platform for live AI digital twins on video calls. Users upload photo + record voice + add knowledge. An AI avatar attends Zoom/WhatsApp calls as them in real-time.

## Your Domain
Everything inside `phantm/frontend/` and `phantm/backend/`.

## COMPLETED: Backend MVP (Steps 1-2)

Backend routers, models, config, database, and middleware are built:
- Auth router: register, login, me (Supabase auth)
- Personas router: CRUD + photo/voice upload endpoints
- Sessions router: create, list, end sessions
- Pydantic models: users, personas, sessions, knowledge, base enums
- Config: pydantic-settings with all env vars
- Database: Supabase client + initial migration SQL
- Middleware: JWT auth middleware

NOTE: Persona photo/voice uploads use placeholder URLs -- R2 storage not wired yet. Wire this when you have time but it is not blocking.

## Current Task: PHASE 2 -- Frontend Scaffold + Dashboard

Now that the backend is ready, build the frontend so users can interact with LiveSelf.

### What to build (in order):

**Step 1: Initialize Next.js 14 project**
```bash
cd phantm/frontend
npx create-next-app@14 . --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
```
- Install Shadcn/ui: npx shadcn@latest init
- Install extra deps: npm install zustand @tanstack/react-query lucide-react
- Install fonts: Sora (UI) and JetBrains Mono (data/code)
- Create the base layout with the Ghost Glass dark theme

**Step 2: API client + auth**
- Create `src/lib/api.ts` -- typed fetch wrapper for backend calls
- Create `src/lib/auth.ts` -- login, register, logout, getUser functions
- Store JWT in httpOnly cookie (NOT localStorage)
- Create auth context/provider with Zustand

**Step 3: Marketing landing page**
Route group: `src/app/(marketing)/`
- Build landing page at `/` with:
  - Hero section: "Show up without showing up." headline + 2 CTAs (Get Started, Watch Demo)
  - "How it works" section: 3 steps (Upload Photo, Record Voice, Go Live)
  - "Built with" tech stack section (logos/names of the open-source tools)
  - Footer with GitHub link
- NO emojis, NO stock photos, NO decorative icons
- Use the Ghost Glass dark design system (see below)

**Step 4: Auth pages**
- `/auth/login` -- email + password form
- `/auth/register` -- email + password + display name form
- Both should redirect to `/app/dashboard` on success
- Clean, minimal design with Ghost Glass styling

**Step 5: App dashboard layout + pages**
Route group: `src/app/(app)/`
- App layout with sidebar navigation (Dashboard, Setup, Sessions)
- `/app/dashboard` -- persona cards + session stats + "Go Live" button
- `/app/setup` -- 3-step wizard:
  - Step 1: Upload face photo (drag & drop)
  - Step 2: Record voice sample (MediaRecorder API, 10-30s)
  - Step 3: Add knowledge (Q&A pairs form)
- `/app/sessions` -- list of past sessions with duration + exchange count

**Step 6: Live session page**
- `/app/sessions/live/[id]` -- the live session screen
- Status HUD showing: connection status, latency, frame rate
- Avatar video feed area (placeholder for now -- virtual cam output)
- "End Session" button
- This page will eventually connect to the engine WebSocket

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
- Cards: glass-morphism effect (backdrop-blur, semi-transparent backgrounds)
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
phantm/frontend/        # Everything here (you create this)
phantm/backend/         # Everything here (Steps 1-2 done)
  app/
    main.py             # EXISTS - FastAPI app
    config.py           # EXISTS - pydantic-settings
    routers/            # EXISTS - auth, personas, sessions
    models/             # EXISTS - base, users, personas, sessions, knowledge
    services/           # EXISTS (empty)
    middleware/          # EXISTS - auth middleware
    database/           # EXISTS - supabase client + migration
```

## Reference
- Full spec: phantm_engineering_bible.html (Sections 01-09, 13-17)
- The old name was "Phantm" in the docs -- project is now "LiveSelf"
- Wireframes are in Section 14 of the engineering bible
- Component list (36 components) is in Section 13
- API endpoints (full 47) are in Section 05 -- but you only build the 12 MVP ones for now
