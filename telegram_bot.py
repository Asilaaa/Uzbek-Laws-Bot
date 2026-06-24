from __future__ import annotations

import asyncio
import html
import logging
import os
import time
import uuid

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag import (
    build_messages,
    get_source_links,
    save_answer_record,
    save_feedback,
    search_chunks,
    stream_completion,
    summarize_sources,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
STREAM_EDIT_INTERVAL_SECONDS = float(os.getenv("STREAM_EDIT_INTERVAL_SECONDS", "1.0"))
STREAM_MIN_CHARS_PER_EDIT = int(os.getenv("STREAM_MIN_CHARS_PER_EDIT", "80"))
TELEGRAM_MAX_TEXT = 4000


def trim_for_telegram(text: str, limit: int = TELEGRAM_MAX_TEXT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n\n[truncated]"


def build_streaming_text(answer_text: str) -> str:
    cleaned = answer_text.strip()
    if not cleaned:
        return "⏳ Generating answer..."
    return f"⚖️ Answer\n\n{cleaned}\n\n⏳ Generating..."


def build_final_text(answer_text: str, source_links) -> str:
    answer_block = html.escape(answer_text.strip() or "No answer generated.")
    parts = [f"<b>⚖️ Answer</b>\n\n{answer_block}"]

    if source_links:
        source_lines = []
        for link in source_links:
            label = html.escape(link.document_name)
            if link.source_url:
                source_lines.append(f'• <a href="{html.escape(link.source_url, quote=True)}">{label}</a>')
            else:
                source_lines.append(f"• {label}")
        parts.append("<b>📎 Sources</b>\n" + "\n".join(source_lines))
    else:
        parts.append("<b>📎 Sources</b>\nNo sources found")

    parts.append("<b>Was this helpful?</b>")
    return "\n\n".join(parts)


def feedback_markup(answer_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("👍", callback_data=f"feedback:up:{answer_id}"),
                InlineKeyboardButton("👎", callback_data=f"feedback:down:{answer_id}"),
            ]
        ]
    )


async def safe_edit_message(message, text: str, reply_markup=None, parse_mode: str | None = None) -> None:
    try:
        await message.edit_text(
            trim_for_telegram(text),
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )
    except BadRequest as exc:
        if "Message is not modified" not in str(exc):
            raise


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send a law question. I will search through my documents and try to answer."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send a question and I will:\n"
        "- Search pgvector\n"
        "- Build context\n"
        "- Generate answer\n"
        "- Collect feedback"
    )


async def question_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()
    if not question:
        await update.message.reply_text("Empty question.")
        return

    if context.chat_data.get("busy"):
        await update.message.reply_text("Still processing previous question.")
        return

    context.chat_data["busy"] = True
    status_message = await update.message.reply_text("Searching legal database...")

    try:
        chunks = await asyncio.to_thread(search_chunks, question)
        messages = build_messages(question, chunks)

        streamed_text = ""
        visible_text = "⏳ Generating answer..."
        last_edit_at = 0.0
        last_len = 0

        await safe_edit_message(status_message, visible_text)

        async for delta in stream_completion(messages):
            streamed_text += delta

            now = time.monotonic()
            time_ok = now - last_edit_at >= STREAM_EDIT_INTERVAL_SECONDS
            growth_ok = len(streamed_text) - last_len >= STREAM_MIN_CHARS_PER_EDIT

            if time_ok or growth_ok:
                visible_text = build_streaming_text(streamed_text)
                await safe_edit_message(status_message, visible_text)
                last_edit_at = now
                last_len = len(streamed_text)

        final_answer = streamed_text.strip() or "No answer generated."
        source_links = await asyncio.to_thread(get_source_links, chunks)
        sources = ", ".join(link.document_name for link in source_links) if source_links else summarize_sources(chunks)
        answer_id = uuid.uuid4().hex

        await asyncio.to_thread(
            save_answer_record,
            answer_id,
            update.effective_chat.id,
            update.effective_user.id if update.effective_user else None,
            update.effective_user.username if update.effective_user else None,
            question,
            final_answer,
            sources,
        )

        final_text = build_final_text(final_answer, source_links)

        await safe_edit_message(
            status_message,
            final_text,
            reply_markup=feedback_markup(answer_id),
            parse_mode="HTML",
        )

    except Exception:
        logger.exception("Handler failed")
        await safe_edit_message(
            status_message,
            "Something went wrong while processing your request."
        )

    finally:
        context.chat_data["busy"] = False


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    parts = query.data.split(":")
    if len(parts) != 3:
        await query.answer()
        return

    _, feedback, answer_id = parts

    try:
        await asyncio.to_thread(save_feedback, answer_id, feedback)
        await query.answer("Thanks!")
        await query.edit_reply_markup(None)
    except Exception:
        logger.exception("Feedback failed")
        await query.answer("Failed to save feedback", show_alert=True)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set")

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(feedback_handler, pattern=r"^feedback:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler))

    logger.info("Telegram law bot started")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()