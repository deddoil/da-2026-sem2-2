import asyncio
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message
from dotenv import load_dotenv

from agent import run_analysis

load_dotenv()

TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY", "")

bot = Bot(TELEGRAM_API_KEY)
dp = Dispatcher()

WELCOME = (
    "Привет! Я - ИИ-аналитик данных.\n\n"
    "Пришли датасет (CSV, TSV или Excel), и я как агент сам исследую его: "
    "посчитаю метрики, найду инсайды, построю графики и пришлю отчёт.\n\n"
    "Можешь добавить инструкцию - что важно (подписью к файлу или сообщением "
    "перед ним). \n\n/help - подробнее."
)

HELP = (
    "Как пользоваться:\n"
    f"1) Пришли файл CSV/TSV/XLSX/XLS\n"
    "2) По желанию добавь инструкцию - подписью к файлу или сообщением перед ним.\n"
    "3) Подожди пару минут - пришлю отчёт и графики.\n\n"
    "Внутри: модель не пересказывает готовые цифры, а сама пишет и выполняет "
    "Python в песочнице (pandas, matplotlib) и итеративно строит анализ.\n\n"
    "Защита: инструкция и содержимое файла считаются недоверенными "
    "данными; попытки prompt-injection отслеживаются, код выполняется без "
    "доступа к секретам бота."
)


@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer(WELCOME)


@dp.message(Command("help"))
async def on_help(message: Message):
    await message.answer(HELP)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())