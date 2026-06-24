# Scout Auto OS — Pre-Deploy Checklist (Hetzner)

## A. Local pre-flight (before server)

- [ ] `paper_mode: true` final validation run (`python scout_auto_os/main.py --once --fast-scan`)
- [ ] Dashboard opens: `python -m streamlit run scout_auto_os/dashboard/app.py`
- [ ] Slot1 / Slot2 EMPTY|OCCUPIED displays correctly
- [ ] Bot Stop button sets `data/bot_control.json` kill_switch (loop keeps monitoring)
- [ ] Force Close button queues `manual_override.json` event
- [ ] 2 empty slots → one scan can fill up to 2 distinct A6 symbols
- [ ] `manual_lock` symbol skipped in TOP5 entry
- [ ] MANUAL position never auto-exited (`check_exits` skip)
- [ ] `logs/orders.csv` path exists (live only)
- [ ] `.env` filled from `.env.example` (NOT committed)

## B. Config (scout_auto_os/config.yaml)

- [ ] `paper_mode: false` only when ready for real 5 USDT orders
- [ ] `execution.order_size_usdt: 5`
- [ ] `position.max_long_slots: 2`
- [ ] `position.meta_rotation: false`
- [ ] `risk.kill_switch: false` (runtime via dashboard OK)
- [ ] `live_data.enabled: true`

## C. Binance account

- [ ] Futures API key created (Trade only, NO withdraw)
- [ ] IP whitelist configured (Hetzner IP after provision)
- [ ] USDT-M wallet funded (minimum 20+ USDT buffer)
- [ ] One-way position mode confirmed
- [ ] Testnet dry-run if available

## D. Hetzner server

- [ ] Python 3.11+ installed
- [ ] `pip install pyyaml websocket-client streamlit`
- [ ] Project copied to `/opt/scout` or similar
- [ ] `systemd` service for `main.py` (24h loop)
- [ ] `systemd` or `screen` for Streamlit dashboard
- [ ] Firewall: dashboard port 8501 restricted to your IP
- [ ] Log rotation: `logs/auto_os/`, `logs/live/`, `logs/orders.csv`

## E. First real session (5 USDT × 2 slots)

- [ ] Start with `paper_mode: false` and ONE slot only (set max_long_slots: 1) for first hour
- [ ] Confirm `logs/orders.csv` shows FILLED entries
- [ ] Confirm exchange position matches DB
- [ ] Test manual close from dashboard → exchange flat
- [ ] Test external manual close on exchange → CLOSED_BY_USER within 30s
- [ ] Enable Slot2 after Slot1 stable

## F. Monitoring

- [ ] Telegram alerts enabled (`alerts.telegram: true` + env vars)
- [ ] Daily report generates at 23:00 KST
- [ ] `engine_status.json` shows `running`
- [ ] `execution_api.json` updates every 30s

## G. Emergency

- [ ] Dashboard → Bot Stop
- [ ] Or set `bot_control.json`: `"kill_switch": true`
- [ ] Or Binance app manual close all positions
- [ ] Revoke API key if compromised
