# Scout Auto OS — Live V1

A6 frozen Long auto-trading stack for Binance Futures (paper / live).

## Hostinger Docker Manager — Compose from URL

### 1. Repository

- GitHub: `https://github.com/keemyungseo/scout-auto-os`
- Compose file (raw URL):

```
https://raw.githubusercontent.com/keemyungseo/scout-auto-os/main/docker-compose.yml
```

### 2. Hostinger setup

1. Open **Hostinger** → **Docker Manager** → **Compose from URL**
2. Paste the raw `docker-compose.yml` URL above
3. Set **Project name**: `scout-auto-os`
4. Before deploy, create `.env` on the server (see below) — **never commit `.env`**

If Hostinger requires a Git repository URL instead of raw compose:

- Repository: `https://github.com/keemyungseo/scout-auto-os`
- Branch: `main`
- Compose path: `docker-compose.yml`

### 3. Environment (`.env`)

Copy `.env.example` → `.env` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `BINANCE_API_KEY` | LIVE | Binance Futures API key (trade only) |
| `BINANCE_API_SECRET` | LIVE | API secret |
| `TRADE_SIZE` | yes | Order size in USDT (default `7`) |
| `LEVERAGE` | yes | Futures leverage (default `3`) |
| `MODE` | yes | `LIVE` or `PAPER` |
| `REPORT_TIME` | no | Daily report hour KST, e.g. `08:00` |
| `TELEGRAM_BOT_TOKEN` | no | Alert bot token |
| `TELEGRAM_CHAT_ID` | no | Alert chat ID |
| `SCOUT_ADMIN_PASSWORD_HASH` | Command Center | bcrypt hash — see [Security](#command-center-security) |
| `SCOUT_COOKIE_SECURE` | no | `true` when served over HTTPS |

### Command Center security

Generate admin password hash (never commit plaintext):

```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_STRONG_PASSWORD', bcrypt.gensalt()).decode())"
```

Add to `.env`:

```
SCOUT_ADMIN_PASSWORD_HASH=$2b$12$...
```

Command Center (`command-center` service) requires this variable. Sessions expire after 30 minutes (sliding). Five failed logins lock the IP for 15 minutes.

### 4. Volumes (persisted on host)

| Host path | Container | Purpose |
|-----------|-----------|---------|
| `./data` | `/app/data` | SQLite DB, runtime config, bot control JSON |
| `./logs` | `/app/logs` | Trades CSV, live WS logs, A6 training cache |

### 5. Local test (before Hostinger)

```bash
cd scout_auto_os
cp .env.example .env
# edit .env with your keys
docker compose up --build -d
docker compose logs -f scout-app
```

### 6. Architecture

```
entrypoint.py
  → load .env
  → patch A6 research paths
  → write /app/data/config.runtime.yaml
  → scout_auto_os.main (5m scan + 30s position loop)
```

- **2 slots** max (`max_long_slots: 2`)
- **No meta rotation**
- **manual_lock** positions are never auto-exited
- Dashboard (optional): `python -m streamlit run dashboard/app.py`

### 7. Emergency stop

1. Hostinger: stop `scout-app` container
2. Or set `kill_switch: true` in `/app/data/bot_control.json` (via `./data/bot_control.json` on host)
3. Or close positions manually on Binance

### 8. Security

- Never commit `.env`, API keys, or `data/` / `logs/`
- Use Binance IP whitelist for your Hostinger server IP
- API key: **Trade** permission only, **no withdraw**

See also: `DEPLOY_CHECKLIST.md`
