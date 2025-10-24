import os
import time
import logging
import sqlite3
import re
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime, timezone
import asyncio  # added for Python 3.14 event loop

import requests
from dotenv import load_dotenv
from telegram import Update, BotCommand, InputFile
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# ========= Load env =========
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_FALLBACKS = [m.strip() for m in os.getenv("OPENAI_FALLBACKS", "gpt-4o-mini,gpt-4o,o4-mini").split(",") if m.strip()]
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))

ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))

# ========= Endless constants =========
ENDLESS_TICKER = "Endless"
ENDLESS_CONTRACT = "0x7abf606f3c1fbef2eb02faeab52b12e325224444"
ENDLESS_X = "https://x.com/endlessbnb"
ENDLESS_TG = "https://t.me/endless_bsc"
ENDLESS_WEBSITE = "https://endlessbnb.com"

WELCOME_IMAGE_PATH = Path("assets/welcome.jpg")
INFO_IMAGE_PATH = Path("assets/info.jpg")

# ========= Rate limits for groups =========
GROUP_CMD_COOLDOWN = 2
GROUP_USER_PER_MIN_LIMIT = 20

# ========= Logging =========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("endless-tg-bot")

# ========= Database =========
DB_PATH = os.getenv("DB_PATH", "endless_bot.db")

def db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def init_db():
    con = db()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        role TEXT,
        content TEXT,
        created_at INTEGER
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        last_request_at INTEGER,
        persona TEXT,
        nickname TEXT,
        timezone TEXT,
        interests TEXT,
        is_holder INTEGER DEFAULT 0,
        holder_since TEXT,
        notes TEXT,
        last_cmd_at INTEGER,
        last_minute_start INTEGER,
        replies_in_minute INTEGER DEFAULT 0,
        points INTEGER DEFAULT 0
        -- column welcomed will be added by migrate_db if missing
    )""")
    con.commit()
    con.close()

def migrate_db():
    con = db()
    cur = con.cursor()
    cur.execute("PRAGMA table_info(users)")
    cols = {r[1] for r in cur.fetchall()}
    if "welcomed" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN welcomed INTEGER DEFAULT 0")
        con.commit()
    con.close()

init_db()
migrate_db()

# ========= Helpers =========
def now_ts() -> int:
    return int(time.time())

def upsert_user(user_id: int):
    con = db()
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO users (user_id, last_request_at) VALUES (?, ?)", (user_id, 0))
    con.commit()
    con.close()

def mark_welcomed(user_id: int):
    con = db()
    cur = con.cursor()
    cur.execute("UPDATE users SET welcomed = 1 WHERE user_id = ?", (user_id,))
    con.commit()
    con.close()

def is_welcomed(user_id: int) -> bool:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT welcomed FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    return bool(row and row[0])

def save_message(user_id: int, role: str, content: str):
    con = db()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (user_id, role, content, now_ts()),
    )
    con.commit()
    con.close()

def get_history(user_id: int, limit: int = 12):
    con = db()
    cur = con.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    )
    rows = list(reversed(cur.fetchall()))
    con.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def set_persona(user_id: int, persona: Optional[str]):
    con = db()
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO users (user_id, last_request_at, persona)
        VALUES (?, COALESCE((SELECT last_request_at FROM users WHERE user_id = ?), 0), ?)
    """, (user_id, user_id, persona))
    con.commit()
    con.close()

def get_persona(user_id: int) -> Optional[str]:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT persona FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    return row[0] if row and row[0] else None

def set_cmd_timestamps(user_id: int):
    now = now_ts()
    con = db()
    cur = con.cursor()
    cur.execute("SELECT last_minute_start, replies_in_minute FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        upsert_user(user_id)
        last_minute_start = now
        replies = 0
    else:
        last_minute_start, replies = row
        if not last_minute_start:
            last_minute_start = now
        if now - last_minute_start >= 60:
            last_minute_start = now
            replies = 0
    replies = (replies or 0) + 1
    cur.execute("""
        UPDATE users
        SET last_cmd_at = ?, last_minute_start = ?, replies_in_minute = ?
        WHERE user_id = ?
    """, (now, last_minute_start, replies, user_id))
    con.commit()
    con.close()
    return replies, last_minute_start

def check_group_limits(user_id: int) -> Tuple[bool, str]:
    con = db()
    cur = con.cursor()
    cur.execute("SELECT last_cmd_at, last_minute_start, replies_in_minute FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    now = now_ts()
    if not row:
        return True, ""
    last_cmd_at, last_minute_start, replies = row
    if last_cmd_at and now - last_cmd_at < GROUP_CMD_COOLDOWN:
        return False, "Please wait a few seconds before the next command."
    if last_minute_start and now - last_minute_start < 60 and replies and replies >= GROUP_USER_PER_MIN_LIMIT:
        return False, "Easy there. Rate limit reached. Try again in a minute."
    return True, ""

# ========= Endless System Prompt =========
def endless_system_prompt() -> str:
    return f"""
You are Endless AI, the official community helper for Endless on BNB Chain.
Be friendly, clear, and helpful. You can chat about general topics, tech, safety, community, and culture.
NEVER discuss, compare, or quote prices or contract addresses for any token other than Endless.

Brand anchors:
Project website: {ENDLESS_WEBSITE}
Ticker: {ENDLESS_TICKER}
Contract address: {ENDLESS_CONTRACT}
Official X: {ENDLESS_X}
Official Telegram: {ENDLESS_TG}

Reply policy:
In groups, be concise and useful.
In DMs, you may be more detailed if helpful.
Keep replies up to date to the best of your knowledge. If unsure, say so briefly.
No financial advice. No profanity. No hype. Stay positive and meme aware.

Always use the contract exactly as provided when needed.
""".strip()

# ========= OpenAI chat with fallback =========
def _openai_http(messages: List[dict], model: str) -> Tuple[bool, str]:
    if not OPENAI_API_KEY:
        return True, "AI chat is not configured yet. Use /about /links /price meanwhile."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 700}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
        if r.status_code != 200:
            try:
                j = r.json()
                err_text = str(j)
            except Exception:
                err_text = r.text
            return False, err_text
        data = r.json()
        return True, data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return False, str(e)

def openai_chat(messages: List[dict]) -> str:
    models = [OPENAI_MODEL] + [m for m in OPENAI_FALLBACKS if m and m != OPENAI_MODEL]
    last_error = ""
    for m in models:
        ok, result = _openai_http(messages, m)
        if ok and isinstance(result, str) and result.strip():
            return result.strip()
        last_error = result
    if "insufficient_quota" in last_error.lower() or "429" in last_error:
        return "AI is temporarily unavailable due to quota. Commands still work: /price /links /about /faq"
    if "model" in last_error.lower() and "not found" in last_error.lower():
        return "Model not available. Set OPENAI_MODEL=gpt-4o-mini in .env and restart."
    return "AI service hit an error. Please try again soon."

# ========= Price helpers =========
def shorten_ca(ca: str) -> str:
    return f"{ca[:8]}…{ca[-4:]}"

def fetch_price_dexscreener(ca: str):
    url = f"https://api.dexscreener.com/latest/dex/tokens/{ca}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        pairs = data.get("pairs") or []
        if not pairs:
            return None
        best = None
        for p in pairs:
            if p.get("chainId") in ("bsc", "bnb"):
                best = p
                break
        if not best:
            best = pairs[0]
        price_usd = float(best.get("priceUsd") or 0)
        change_24h = float((best.get("priceChange") or {}).get("h24") or 0)
        liquidity_usd = float((best.get("liquidity") or {}).get("usd") or 0)
        fdv = float(best.get("fdv") or 0)
        updated_ms = int(best.get("updatedAt") or 0)
        updated_dt = datetime.fromtimestamp(updated_ms / 1000, tz=timezone.utc) if updated_ms else datetime.now(timezone.utc)
        return {"price": price_usd, "change24": change_24h, "liquidity": liquidity_usd, "mcap": fdv, "updated": updated_dt}
    except Exception:
        return None

def fetch_price_coingecko(ca: str):
    url = "https://api.coingecko.com/api/v3/simple/token_price/binance-smart-chain"
    params = {"contract_addresses": ca, "vs_currencies": "usd", "include_24hr_change": "true", "include_market_cap": "true"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        info = data.get(ca.lower())
        if not info:
            return None
        price = float(info.get("usd") or 0)
        change24 = float(info.get("usd_24h_change") or 0)
        mcap = float(info.get("usd_market_cap") or 0)
        return {"price": price, "change24": change24, "liquidity": 0.0, "mcap": mcap, "updated": datetime.now(timezone.utc)}
    except Exception:
        return None

def get_endless_price():
    data = fetch_price_dexscreener(ENDLESS_CONTRACT)
    if not data:
        data = fetch_price_coingecko(ENDLESS_CONTRACT)
    return data

def format_group_price(d):
    t = datetime.now(timezone.utc)
    mins = int((t - d["updated"]).total_seconds() // 60) if d["updated"] else 0
    when = "just now" if mins == 0 else f"{mins} min ago"
    price = f"{d['price']:.6f}"
    ch = d["change24"]
    ch_text = f"+{ch:.2f}%" if ch >= 0 else f"{ch:.2f}%"
    return (
        f"<b>{ENDLESS_TICKER} price</b> <code>{price} USD</code> "
        f"• 24h {ch_text} • {when} "
        f"• CA <code>{shorten_ca(ENDLESS_CONTRACT)}</code>"
    )

def format_dm_price(d):
    t = datetime.now(timezone.utc)
    mins = int((t - d["updated"]).total_seconds() // 60) if d["updated"] else 0
    when = "just now" if mins == 0 else f"{mins} min ago"
    price = f"{d['price']:.6f}"
    ch = d["change24"]
    ch_text = f"+{ch:.2f}%" if ch >= 0 else f"{ch:.2f}%"
    lines = [
        f"<b>{ENDLESS_TICKER} on BNB Chain</b>",
        f"Price: <code>{price} USD</code>",
        f"24h Change: <code>{ch_text}</code>",
        (f"Market Cap: <code>{d['mcap']:.0f} USD</code>" if d["mcap"] else "Market Cap: <i>n/a</i>"),
        (f"Liquidity: <code>{d['liquidity']:.0f} USD</code>" if d["liquidity"] else "Liquidity: <i>n/a</i>"),
        f"Updated: <i>{when}</i>",
        f"Contract: <code>{ENDLESS_CONTRACT}</code>",
        f"Website: <a href=\"{ENDLESS_WEBSITE}\">{ENDLESS_WEBSITE}</a>",
        f"Links: <a href=\"{ENDLESS_X}\">X</a> • <a href=\"{ENDLESS_TG}\">Telegram</a>",
    ]
    return "\n".join(lines)

# ========= Safety helpers =========
def violates_policy(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return ("seed phrase" in lowered) or ("private key" in lowered)

def mentions_other_tokens(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if "endless" in lowered:
        return False
    if "contract" in lowered or "ca " in lowered or "0x" in lowered:
        return True
    if "price" in lowered or "chart" in lowered:
        return True
    if "$" in text:
        return True
    return False

# ========= Reply helper =========
async def reply_html(update: Update, text: str):
    await update.message.reply_text(text, parse_mode="HTML", disable_web_page_preview=True)

def is_group_chat(update: Update) -> bool:
    return update.effective_chat.type in ("group", "supergroup")

# ========= Commands =========
async def send_first_time_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if is_welcomed(user_id):
        return
    caption = (
        f"<b>Welcome to Endless</b>\n"
        f"Memes, community, and gaming on BNB Chain.\n\n"
        f"Quick commands:\n"
        f"• <code>/about</code> • <code>/links</code> • <code>/price</code> • <code>/faq</code>\n\n"
        f"Website: <a href=\"{ENDLESS_WEBSITE}\">{ENDLESS_WEBSITE}</a>\n"
        f"X: <a href=\"{ENDLESS_X}\">{ENDLESS_X}</a>\n"
        f"Telegram: <a href=\"{ENDLESS_TG}\">{ENDLESS_TG}</a>\n"
        f"Contract: <code>{ENDLESS_CONTRACT}</code>"
    )
    try:
        if WELCOME_IMAGE_PATH.exists():
            with open(WELCOME_IMAGE_PATH, "rb") as f:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=InputFile(f), caption=caption, parse_mode="HTML")
        else:
            await reply_html(update, caption)
        mark_welcomed(user_id)
    except Exception as e:
        logger.warning("Welcome send failed: %s", e)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update.effective_user.id)
    await send_first_time_welcome(update, context)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_group_chat(update):
        await reply_html(update, "Try <code>/about</code> <code>/links</code> <code>/price</code> <code>/meme</code> <code>/faq</code>. For deep help, DM me.")
    else:
        await reply_html(update,
            "I help with Endless info and safety on BNB Chain.\n"
            "Commands:\n"
            "<code>/start</code> <code>/about</code> <code>/links</code> <code>/price</code> <code>/meme</code> <code>/faq</code> <code>/verify</code> <code>/rank</code> <code>/info</code>"
        )

async def cmd_about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"<b>About Endless</b>\n"
        f"Endless unites memes, community, and gaming on BNB Chain. "
        f"The goal is to blend fun and utility so participation feels rewarding.\n\n"
        f"Website: <a href=\"{ENDLESS_WEBSITE}\">{ENDLESS_WEBSITE}</a>\n"
        f"X: <a href=\"{ENDLESS_X}\">{ENDLESS_X}</a> • TG: <a href=\"{ENDLESS_TG}\">{ENDLESS_TG}</a>\n"
        f"Contract: <code>{ENDLESS_CONTRACT}</code>"
    )
    await reply_html(update, msg)

async def cmd_links(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await reply_html(update,
        f"<b>Official Links</b>\n"
        f"Website: <a href=\"{ENDLESS_WEBSITE}\">{ENDLESS_WEBSITE}</a>\n"
        f"X: <a href=\"{ENDLESS_X}\">{ENDLESS_X}</a>\n"
        f"Telegram: <a href=\"{ENDLESS_TG}\">{ENDLESS_TG}</a>\n"
        f"Contract: <code>{ENDLESS_CONTRACT}</code>"
    )

async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    is_group = is_group_chat(update)
    if is_group:
        ok, reason = check_group_limits(user_id)
        if not ok:
            await reply_html(update, reason)
            return
        set_cmd_timestamps(user_id)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    data = get_endless_price()
    if not data:
        await reply_html(update, "Price is temporarily unavailable. Please try again soon.")
        return
    if is_group:
        await reply_html(update, format_group_price(data))
    else:
        await reply_html(update, format_dm_price(data))

async def cmd_meme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    one_liners = [
        "Endless cooking.",
        "Endless fun only.",
        "Build with conviction. Community first.",
        "If it is not fun it is not Endless.",
    ]
    await reply_html(update, f"🧠 {one_liners[int(time.time()) % len(one_liners)]}")

async def cmd_faq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>Common Questions</b>\n"
        "• What is Endless? Meme plus utility on BNB Chain.\n"
        "• How to follow? See /links.\n"
        "• How to verify? Use /verify when enabled.\n"
        "• Safety. Admins never ask for money or keys."
    )
    await reply_html(update, text)

async def cmd_verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    con = db()
    cur = con.cursor()
    cur.execute("UPDATE users SET is_holder = 1, holder_since = ? WHERE user_id = ?", (datetime.utcnow().isoformat(), user_id))
    con.commit()
    con.close()
    await reply_html(update, "Holder status saved. Full verification will arrive later.")

async def cmd_rank(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    con = db()
    cur = con.cursor()
    cur.execute("SELECT points FROM users WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    pts = row[0] if row else 0
    con.close()
    await reply_html(update, f"Your community points: <b>{pts}</b>")

async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    card = (
        f"<b>Endless — Quick Info</b>\n"
        f"Memes, community, and gaming on BNB Chain.\n\n"
        f"Website: <a href=\"{ENDLESS_WEBSITE}\">{ENDLESS_WEBSITE}</a>\n"
        f"X: <a href=\"{ENDLESS_X}\">{ENDLESS_X}</a> • "
        f"TG: <a href=\"{ENDLESS_TG}\">{ENDLESS_TG}</a>\n"
        f"Contract: <code>{ENDLESS_CONTRACT}</code>\n\n"
        f"Try: <code>/about</code> • <code>/links</code> • <code>/price</code> • <code>/faq</code>"
    )
    try:
        if INFO_IMAGE_PATH.exists():
            with open(INFO_IMAGE_PATH, "rb") as f:
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=InputFile(f),
                    caption=card,
                    parse_mode="HTML",
                )
        else:
            await update.message.reply_text(card, parse_mode="HTML", disable_web_page_preview=True)
    except Exception:
        await update.message.reply_text(card, parse_mode="HTML", disable_web_page_preview=True)

# ========= Group trigger helper =========
def _strip_group_trigger(text: str) -> Tuple[bool, str]:
    if not text:
        return False, text
    s = text.lstrip()
    m = re.match(r"(?i)^(endless\s*ai)\b[\s,:-]*", s)
    if m:
        return True, s[m.end():].lstrip()
    return False, text

# ========= Free text handler =========
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    upsert_user(update.effective_user.id)
    text = (update.message.text or "").strip()

    if is_group_chat(update):
        triggered, stripped = _strip_group_trigger(text)
        reply_to_bot = bool(
            getattr(update.message, "reply_to_message", None)
            and getattr(update.message.reply_to_message, "from_user", None)
            and update.message.reply_to_message.from_user.id == context.bot.id
        )
        if not (triggered or reply_to_bot):
            return
        if triggered:
            text = stripped if stripped else "help"

    if violates_policy(text):
        await reply_html(update, "Stay safe. Never share seed phrases or private keys. Only trust the main group and official X.")
        return

    if mentions_other_tokens(text):
        await reply_html(update,
            "I can chat about general topics or Endless.\n"
            "I cannot discuss prices or contracts for other tokens.\n"
            f"For Endless, try <code>/price</code> or visit <a href=\"{ENDLESS_X}\">updates</a>."
        )
        return

    await send_first_time_welcome(update, context)

    persona = get_persona(update.effective_user.id) or endless_system_prompt()
    # Group vs DM behavior: short in groups, flexible in DMs
    if is_group_chat(update):
        persona += "\nKeep replies short and concise for group chats."
    else:
        persona += "\nYou may give detailed or short replies in DM depending on what the user wants."

    messages: List[dict] = [{"role": "system", "content": persona}]
    for m in get_history(update.effective_user.id, limit=8):
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": text})

    if is_group_chat(update):
        ok, reason = check_group_limits(update.effective_user.id)
        if not ok:
            await reply_html(update, reason)
            return
        set_cmd_timestamps(update.effective_user.id)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    reply = openai_chat(messages)

    # No character truncation here (group replies are guided to be short via persona)

    save_message(update.effective_user.id, "user", text)
    save_message(update.effective_user.id, "assistant", reply if isinstance(reply, str) else str(reply))
    await reply_html(update, reply if isinstance(reply, str) else str(reply))

# ========= Startup hooks =========
async def post_init(app: Application):
    commands = [
        BotCommand("start", "Welcome"),
        BotCommand("help", "What I can do"),
        BotCommand("about", "About Endless"),
        BotCommand("links", "Official links and contract"),
        BotCommand("price", "Endless price"),
        BotCommand("meme", "One liner"),
        BotCommand("faq", "Common questions"),
        BotCommand("verify", "Mark holder stub"),
        BotCommand("rank", "Your points"),
        BotCommand("info", "Project quick card"),
    ]
    try:
        await app.bot.set_my_commands(commands)
        await app.bot.set_my_short_description("Endless AI — community helper on BNB Chain")
        await app.bot.set_my_description(
            "Endless AI welcomes users, answers Endless questions, shows price snippets, "
            "and keeps chats safe and fun. Official links in /links."
        )
    except Exception as e:
        logger.warning("Unable to set commands or description: %s", e)

# ========= App wiring =========
def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("Missing TELEGRAM_TOKEN. Set it in .env")

    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("links", cmd_links))
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("meme", cmd_meme))
    app.add_handler(CommandHandler("faq", cmd_faq))
    app.add_handler(CommandHandler("verify", cmd_verify))
    app.add_handler(CommandHandler("rank", cmd_rank))
    app.add_handler(CommandHandler("info", cmd_info))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Python 3.14 ensure an event loop exists
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    app.run_polling()

if __name__ == "__main__":
    main()
