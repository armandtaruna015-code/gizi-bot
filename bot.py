import os
import json
import base64
import sqlite3
import tempfile
from datetime import datetime, date
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from openai import OpenAI

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not BOT_TOKEN or not OPENAI_KEY:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN dan OPENAI_API_KEY di .env")

client = OpenAI(api_key=OPENAI_KEY)
DB_PATH = "gizi_bot.db"


# ---------- DB ----------
def db():
    return sqlite3.connect(DB_PATH)

def init_db():
    con = db()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
      user_id INTEGER PRIMARY KEY,
      weight_kg REAL,
      height_cm REAL,
      goal TEXT,
      target_kcal REAL,
      target_protein REAL,
      updated_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS food_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      ts TEXT NOT NULL,
      source TEXT NOT NULL,
      raw_input TEXT NOT NULL,
      calories REAL,
      protein REAL,
      carbs REAL,
      fat REAL,
      diet_rating TEXT,
      advice TEXT
    )
    """)
    con.commit()
    con.close()

def upsert_user(user_id: int, weight_kg: float, height_cm: float, goal: str):
    base = weight_kg * 30.0
    if goal == "cut":
        target_kcal = max(1200.0, base - 400.0)
    elif goal == "bulk":
        target_kcal = base + 300.0
    else:
        target_kcal = base

    target_protein = weight_kg * 1.6

    con = db()
    cur = con.cursor()
    cur.execute("""
      INSERT INTO users (user_id, weight_kg, height_cm, goal, target_kcal, target_protein, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(user_id) DO UPDATE SET
        weight_kg=excluded.weight_kg,
        height_cm=excluded.height_cm,
        goal=excluded.goal,
        target_kcal=excluded.target_kcal,
        target_protein=excluded.target_protein,
        updated_at=excluded.updated_at
    """, (user_id, weight_kg, height_cm, goal, target_kcal, target_protein, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    con.close()
    return target_kcal, target_protein

def get_user(user_id: int):
    con = db()
    cur = con.cursor()
    cur.execute("SELECT weight_kg, height_cm, goal, target_kcal, target_protein FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    con.close()
    return row

def log_food(user_id: int, source: str, raw_input: str, result: dict):
    con = db()
    cur = con.cursor()
    cur.execute("""
      INSERT INTO food_log (user_id, ts, source, raw_input, calories, protein, carbs, fat, diet_rating, advice)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        datetime.now().isoformat(timespec="seconds"),
        source,
        raw_input,
        float(result["total"]["calories"]),
        float(result["total"]["protein_g"]),
        float(result["total"]["carbs_g"]),
        float(result["total"]["fat_g"]),
        result.get("diet_rating", "YELLOW"),
        json.dumps(result.get("advice", []), ensure_ascii=False)
    ))
    con.commit()
    con.close()

def sum_today(user_id: int):
    con = db()
    cur = con.cursor()
    today = date.today().isoformat()
    cur.execute("""
      SELECT
        COALESCE(SUM(calories),0),
        COALESCE(SUM(protein),0),
        COALESCE(SUM(carbs),0),
        COALESCE(SUM(fat),0)
      FROM food_log
      WHERE user_id=? AND substr(ts,1,10)=?
    """, (user_id, today))
    row = cur.fetchone()
    con.close()
    return row

def reset_today(user_id: int):
    con = db()
    cur = con.cursor()
    today = date.today().isoformat()
    cur.execute("DELETE FROM food_log WHERE user_id=? AND substr(ts,1,10)=?", (user_id, today))
    con.commit()
    con.close()


# ---------- OpenAI structured output schema ----------
SCHEMA = {
  "name": "food_analysis",
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "items": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "name": {"type": "string"},
            "estimated_portion": {"type": "string"},
            "calories": {"type": "number"},
            "protein_g": {"type": "number"},
            "carbs_g": {"type": "number"},
            "fat_g": {"type": "number"}
          },
          "required": ["name", "estimated_portion", "calories", "protein_g", "carbs_g", "fat_g"]
        }
      },
      "total": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "calories": {"type": "number"},
          "protein_g": {"type": "number"},
          "carbs_g": {"type": "number"},
          "fat_g": {"type": "number"}
        },
        "required": ["calories", "protein_g", "carbs_g", "fat_g"]
      },
      "diet_rating": {"type": "string", "enum": ["GREEN", "YELLOW", "RED"]},
      "advice": {"type": "array", "items": {"type": "string"}},
      "follow_up_questions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["items", "total", "diet_rating", "advice", "follow_up_questions"]
  }
}

def call_ai_text(food_text: str, user_ctx: dict | None = None) -> dict:
    context = user_ctx or {}
    prompt = (
        "Kamu adalah asisten tracking gizi.\n"
        "Estimasi item makanan/minuman dari teks user, hitung total kalori & makro.\n"
        "Kasih diet_rating GREEN/YELLOW/RED (goal 'cut' utamakan defisit & protein).\n"
        "Kasih advice singkat maksimal 3 poin.\n"
        "Kalau porsi/komposisi belum jelas, isi best guess tapi tanya max 2 follow_up_questions.\n"
        f"Context user: {json.dumps(context, ensure_ascii=False)}\n"
        f"Input user: {food_text}\n"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        response_format={"type": "json_schema", "json_schema": SCHEMA}
    )
    return json.loads(resp.output_text)

def call_ai_photo(image_bytes: bytes, caption: str, user_ctx: dict | None = None) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    context = user_ctx or {}
    instruction = (
        "Kamu adalah asisten tracking gizi.\n"
        "Dari foto makanan, identifikasi menu & perkiraan porsi, lalu hitung kalori & makro.\n"
        "Kasih diet_rating GREEN/YELLOW/RED sesuai goal.\n"
        "Kasih advice maksimal 3 poin + saran porsi.\n"
        "Kalau porsi tidak jelas, tanya max 2 follow_up_questions.\n"
        f"Context user: {json.dumps(context, ensure_ascii=False)}\n"
        f"Caption/teks tambahan user: {caption}\n"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        response_format={"type": "json_schema", "json_schema": SCHEMA}
    )
    return json.loads(resp.output_text)


# ---------- Telegram handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Yo by 👋\n"
        "Kirim makanan kamu via teks atau foto.\n\n"
        "Wajibin setup dulu biar ada target:\n"
        "/setup 80 170 cut\n"
        "Format: /setup <BBkg> <TBcm> <cut|maintain|bulk>\n\n"
        "Command:\n"
        "/today - recap hari ini\n"
        "/resetday - hapus log hari ini"
    )

async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        weight = float(context.args[0])
        height = float(context.args[1])
        goal = context.args[2].lower()
        if goal not in ["cut", "maintain", "bulk"]:
            raise ValueError("goal invalid")
    except Exception:
        await update.message.reply_text("Formatnya gini ya by: /setup 80 170 cut")
        return

    user_id = update.effective_user.id
    target_kcal, target_protein = upsert_user(user_id, weight, height, goal)

    await update.message.reply_text(
        f"Oke ke-save ✅\n"
        f"Goal: {goal}\n"
        f"Target harian (estimasi): {target_kcal:.0f} kkal\n"
        f"Protein minimal: {target_protein:.0f} g\n\n"
        "Sekarang kirim makanan kamu (teks/foto)."
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cal, pro, carb, fat = sum_today(user_id)

    user = get_user(user_id)
    if user:
        _, _, goal, target_kcal, target_protein = user
        sisa = target_kcal - cal
        await update.message.reply_text(
            f"Recap hari ini 🍽️\n"
            f"Kalori: {cal:.0f} kkal (sisa {sisa:.0f})\n"
            f"Protein: {pro:.0f} g (target {target_protein:.0f})\n"
            f"Karbo: {carb:.0f} g\n"
            f"Lemak: {fat:.0f} g"
        )
    else:
        await update.message.reply_text(
            f"Recap hari ini 🍽️\n"
            f"Kalori: {cal:.0f} kkal\n"
            f"Protein: {pro:.0f} g\n"
            f"Karbo: {carb:.0f} g\n"
            f"Lemak: {fat:.0f} g\n\n"
            "Biar ada target, jalanin /setup dulu ya by."
        )

async def resetday_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reset_today(user_id)
    await update.message.reply_text("Oke, log hari ini udah dihapus ✅")

def format_ai_result(result: dict) -> str:
    total = result["total"]
    rating = result.get("diet_rating", "YELLOW")
    advice = result.get("advice", [])
    fu = result.get("follow_up_questions", [])

    emoji = {"GREEN": "✅", "YELLOW": "🟡", "RED": "🔴"}.get(rating, "🟡")

    msg = (
        f"{emoji} Diet check: {rating}\n"
        f"Estimasi total:\n"
        f"- Kalori: {total['calories']:.0f} kkal\n"
        f"- Protein: {total['protein_g']:.0f} g\n"
        f"- Karbo: {total['carbs_g']:.0f} g\n"
        f"- Lemak: {total['fat_g']:.0f} g\n"
    )

    if advice:
        msg += "\nSaran:\n" + "\n".join([f"- {a}" for a in advice[:3]])

    if fu:
        msg += "\n\nBiar makin akurat, jawab ini ya:\n" + "\n".join([f"- {q}" for q in fu[:2]])

    return msg

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    user = get_user(user_id)
    user_ctx = None
    if user:
        weight, height, goal, target_kcal, target_protein = user
        user_ctx = {
            "weight_kg": weight,
            "height_cm": height,
            "goal": goal,
            "target_kcal": target_kcal,
            "target_protein": target_protein
        }

    result = call_ai_text(text, user_ctx=user_ctx)
    log_food(user_id, "text", text, result)
    await update.message.reply_text(format_ai_result(result))

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    photo = update.message.photo[-1]
    tg_file = await photo.get_file()

    # temp file yang aman untuk Windows/Linux
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    await tg_file.download_to_drive(custom_path=tmp_path)

    with open(tmp_path, "rb") as f:
        img_bytes = f.read()

    caption = update.message.caption or ""

    user = get_user(user_id)
    user_ctx = None
    if user:
        weight, height, goal, target_kcal, target_protein = user
        user_ctx = {
            "weight_kg": weight,
            "height_cm": height,
            "goal": goal,
            "target_kcal": target_kcal,
            "target_protein": target_protein
        }

    result = call_ai_photo(img_bytes, caption=caption, user_ctx=user_ctx)
    log_food(user_id, "photo", caption or "[photo]", result)
    await update.message.reply_text(format_ai_result(result))


def main():
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setup", setup))
    app.add_handler(CommandHandler("today", today))
    app.add_handler(CommandHandler("resetday", resetday_cmd))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling()

if __name__ == "__main__":
    main()
