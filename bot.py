import asyncio
import os
import shutil
import tempfile

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv

from agent import run_analysis

load_dotenv()

TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY", "")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "3"))

ALLOWED_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm"}


def validate_file(filename, size_bytes):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        return False, (f"Неподдерживаемый формат: {ext or '-'}. "
                       "Поддерживаются CSV, TSV, XLSX, XLS.")
    return True, ""

bot = Bot(TELEGRAM_API_KEY)
dp = Dispatcher()
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

WELCOME = (
    "Привет! Я - ИИ-аналитик данных.\n\n"
    "Пришли датасет (CSV, TSV или Excel), и я как агент сам исследую его: "
    "посчитаю метрики, найду инсайды, построю графики и пришлю отчёт.\n\n"
    "Можешь добавить инструкцию - что важно."
    "\n\n/help - подробнее."
)

HELP = (
    "Как пользоваться:\n"
    f"1) Пришли файл CSV/TSV/XLSX/XLS\n"
    "2) По желанию добавь инструкцию подписью к файлу\n"
    "3) Подожди пару минут - пришлю отчёт и графики.0"
)


@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer(WELCOME)


@dp.message(Command("help"))
async def on_help(message: Message):
    await message.answer(HELP)


@dp.message(F.document)
async def on_document(message: Message):

    doc = message.document
    ok, err = validate_file(doc.file_name, doc.file_size)
    if not ok:
        await message.answer(err)
        return

    instruction = message.caption or ""

    session_dir = tempfile.mkdtemp(prefix="analysis_")
    data_path = os.path.join(session_dir, doc.file_name)
    try:
        await bot.download(doc, destination=data_path)
    except Exception as exc:
        await message.answer(f"Не удалось скачать файл: {exc}")
        shutil.rmtree(session_dir, ignore_errors=True)
        return

    prefix = ""

    status = await message.answer("Анализирую данные… это может занять пару минут.")

    try:

        async with analysis_semaphore:
            result = await asyncio.to_thread(
                run_analysis, data_path, session_dir, instruction)
        await _send_report(message, prefix + result["report"], result["images"])

    except Exception as exc:
        await message.answer(f"Произошла ошибка при анализе: {exc}")
    finally:
        try:
            await status.delete()
        except Exception:
            pass
        shutil.rmtree(session_dir, ignore_errors=True)

@dp.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message):

    await message.answer("Похоже, ты забыл прикрепить файл")


async def _send_report(message: Message, report, images):
    #Отчёт режем на куски (лимит Telegram), графики шлём отдельными фото
    for chunk in _split(report, 4000):
        await message.answer(chunk)
    for image in images:
        try:
            caption = (image.get("title") or "")[:1024] or None
            await message.answer_photo(FSInputFile(image["path"]), caption=caption)
        except Exception as exc:
            pass


def _split(text, size):
    text = text or "(пустой отчёт)"
    return [text[i:i + size] for i in range(0, len(text), size)]


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())