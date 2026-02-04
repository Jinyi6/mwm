# AI Town Lite

轻量 AI Town 原型（Vite + React + FastAPI + SQLite）。

## Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=YOUR_BASE_URL  # optional
export SQLITE_PATH=./data/app.db
uvicorn apps.api.main:app --reload
```

## Frontend

```bash
cd apps/web
npm install
npm run dev
```

## Load Test (k6)

```bash
k6 run scripts/k6.js
```

## Notes
- SQLite file default: `./data/app.db`
- World model updates are transactional with version check (409 on conflict)
- 中文文档见 `docs/README.md`
