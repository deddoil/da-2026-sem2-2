import os
import sys
import re
import json
import uuid
import textwrap
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

log = logging.getLogger("agent")

# Модель и настройки
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
PYTHON_TIMEOUT = int(os.getenv("AGENT_PYTHON_TIMEOUT", "180"))

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY не задан")
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://t.me/your_bot",
                "X-Title": "Data Analyst Telegram Bot",
            },
        )
    return _client


RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Executes Python code for data analysis. "
            "You may create at most one matplotlib figure per call. "
            "To save a figure, use "
            "plt.savefig(os.environ['FIGURE_PATH'], dpi=120, bbox_inches='tight') "
            "and then plt.close(). "
            "pandas, numpy, matplotlib, seaborn are available. "
            "The dataset is located at os.environ['DATA_PATH'] and is also available "
            "in the variable DATA_PATH."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Can be multiline.",
                }
            },
            "required": ["code"],
        },
    },
}


# --- Парсер текстовых tool call'ов (для моделей без native function calling) ---

# Основной вариант: <tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>
_TEXT_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*"
    r"<function\s*=\s*(?P<fn>\w+)\s*>\s*"
    r"<parameter\s*=\s*(?P<arg>\w+)\s*>\s*"
    r"(?P<val>.*?)"
    r"\s*</parameter>\s*"
    r"</function>\s*"
    r"</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

# Запасной вариант: без внешнего <tool_call>...</tool_call>
_TEXT_TOOL_CALL_RE_LOOSE = re.compile(
    r"<function\s*=\s*(?P<fn>\w+)\s*>\s*"
    r"<parameter\s*=\s*(?P<arg>\w+)\s*>\s*"
    r"(?P<val>.*?)"
    r"\s*</parameter>\s*"
    r"</function>",
    re.DOTALL | re.IGNORECASE,
)

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n?|\n?```\s*$")


def _strip_code_fences(code: str) -> str:
    # Модели любят заворачивать код в ```python ... ```; убираем обёртку.
    if not code:
        return code
    return _FENCE_RE.sub("", code).strip("\n")


def parse_text_tool_call(content: str):

    if not content:
        return None
    m = _TEXT_TOOL_CALL_RE.search(content) or _TEXT_TOOL_CALL_RE_LOOSE.search(content)
    if not m:
        return None
    fn = m.group("fn")
    arg = m.group("arg")
    if fn != "run_python" or arg != "code":
        return None
    return fn, _strip_code_fences(m.group("val"))


def build_system_prompt(data_path: str) -> str:
    return f"""You are a data analysis agent. Your task is to perform exploratory data analysis (EDA) on a dataset and return a meaningful text report.

You have a single tool available: run_python. You must call it to write and execute Python code. Do not try to guess numbers — compute them with code.

HOW TO CALL THE TOOL:
- If native function calling is available in your environment, use it directly.
- Otherwise, emit the call strictly in this format, with nothing else around it:

  <tool_call>
  <function=run_python>
  <parameter=code>
  ... python code here ...
  </parameter>
  </function>
  </tool_call>

  Inside <parameter=code> put only Python code, no markdown fences like ```python.
  Do NOT wrap that block in commentary if you actually want to call the tool.
  A final report (when the analysis is done) is plain text WITHOUT these tags.

Rules:
- The dataset is located at: {data_path}
- In your code, the variable DATA_PATH is available and equals this path.
- Use pandas, numpy, matplotlib, seaborn.
- In a single run_python call, create at most one matplotlib figure. Save it to the file specified by os.environ['FIGURE_PATH'], for example: plt.savefig(os.environ['FIGURE_PATH'], dpi=120, bbox_inches='tight'), then plt.close().
- If a plot is not needed, do not create a figure.
- After each call, analyze stdout/stderr and decide what to do next.
- Take into account the user's instruction, if provided.
- When the analysis is complete, return the final report as plain text (without JSON and without calling the tool). The report must include the following sections (in Russian): Обзор данных, Ключевые метрики, Тренды и закономерности, Аномалии, Бизнес-выводы, Ограничения.
- Do not make up data. If something is missing, state it in the limitations.
- Write the final report in Russian.

FORMATTING RULES FOR THE FINAL REPORT:
- Do NOT use any Markdown syntax. 
- Never use asterisks (*), double asterisks (**), underscores (_), backticks (`), hash symbols (#), greater-than (>) blockquotes, or hyphens (-) at the start of a line as bullet markers.
- Do not use tables, HTML tags, or any other markup.
- For lists, use a plain hyphen with a space ("- item") only if the list is short and clearly reads better than prose. Otherwise use plain sentences.
- Do not make up data. If something is missing, state it in the limitations section.

PLOT LABELS RULES (very important):
- Column names in the raw data often contain long encoding explanations,
  for example: "Gender(0=Male;1=Female;)", "Religion(0=Christian;1=Jewish;...)",
  "Political_Party(0=Democratic;1=Republican;...)", "Social_Media(10=V10_or_more;)".
- On ANY plot (heatmap, scatter, bar, boxplot, etc.) use SHORT labels for axes and
  legends — only the "human" feature name, without parentheses and without listing
  codes. Rules:
    1) Cut off everything from "(" to the matching ")". So
       "Gender(0=Male;1=Female;)" -> "Gender", "Religion(0=Christian;...)" ->
       "Religion", "Political_Party(0=Democratic;...)" -> "Political_Party".
    2) Replace underscores with spaces: "Party_Hours_per_week" ->
       "Party hours per week", "Home_Town" -> "Home town".
    3) Do not add code values, ranges, percentages, or units in parentheses to
       labels — only the clean feature name.
    4) For categorical features, label the legend with human-readable names, not
       codes. For example, for Gender(0=Male;1=Female;) the legend should show
       "Male" and "Female", not "0" and "1".,,,
- Do this transformation directly in code using a simple function, for example:

    import re
    def short_name(col: str) -> str:
        s = re.sub(r"\(.*?\)", "", str(col)).strip()
        s = s.replace("_", " ").strip()
        return s or str(col)

  and pass `short_name(col)` when setting labels.

- Label axis values for numeric features meaningfully (e.g., "Drinks per week",
  "Hours per week"), not "col1", "col2".
"""


def _read_dataframe(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def get_dataset_summary(data_path: str) -> str:
    try:
        df = _read_dataframe(data_path)
        summary = {
            "shape": list(df.shape),
            "columns": list(map(str, df.columns)),
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
            "missing": {str(c): int(v) for c, v in df.isna().sum().items()},
        }
        num = df.select_dtypes(include=[np.number])
        if not num.empty:
            summary["numeric_describe"] = json.loads(
                num.describe().to_json(force_ascii=False)
            )
        return json.dumps(summary, ensure_ascii=False, default=str, indent=2)
    except Exception as e:
        return f"Не удалось прочитать датасет: {e}"


def execute_python(code: str, session_dir: str, data_path: str) -> Dict[str, Any]:
    session_dir = os.path.abspath(session_dir)
    os.makedirs(session_dir, exist_ok=True)

    fig_path = os.path.join(session_dir, f"figure_{uuid.uuid4().hex}.png")

    env = os.environ.copy()
    env["DATA_PATH"] = os.path.abspath(data_path)
    env["FIGURE_PATH"] = fig_path
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    prefix = textwrap.dedent(
        """
        import os, sys, json, math, re
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        DATA_PATH = os.environ.get('DATA_PATH')
        FIGURE_PATH = os.environ.get('FIGURE_PATH')
        """
    )
    full_code = prefix + "\n" + code

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_code],
            cwd=session_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT,
        )
        stdout = proc.stdout[-8000:] if proc.stdout else ""
        stderr = proc.stderr[-8000:] if proc.stderr else ""
        result_text = (
            f"returncode: {proc.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    except subprocess.TimeoutExpired as e:
        result_text = (
            f"Превышено время выполнения ({PYTHON_TIMEOUT} сек).\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        )
    except Exception as e:
        result_text = f"Ошибка запуска кода: {e}"

    figure = None
    if os.path.exists(fig_path) and os.path.getsize(fig_path) > 0:
        figure = {"path": fig_path, "title": "График"}
        result_text += f"\n\n[Сохранён график: {fig_path}]"
    else:
        if os.path.exists(fig_path):
            try:
                os.remove(fig_path)
            except OSError:
                pass

    return {"text": result_text, "figure": figure}


def _msg_content(msg) -> str:
    # OpenAI SDK возвращает объект; достаём текст безопасно.
    content = getattr(msg, "content", None)
    return content or ""


def run_analysis(
    data_path: str,
    session_dir: str,
    instruction: str = ""
) -> Dict[str, Any]:
    client = get_client()
    data_path = os.path.abspath(data_path)
    session_dir = os.path.abspath(session_dir)
    os.makedirs(session_dir, exist_ok=True)

    dataset_summary = get_dataset_summary(data_path)
    system_prompt = build_system_prompt(data_path)

    user_prompt = f"""Инструкция пользователя: {instruction.strip() or 'нет'}

Сводка по датасету:
{dataset_summary}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    images: List[Dict[str, str]] = []
    report = ""
    iterations = 0

    def _execute_and_append(tool_call_id: str, code: str, step: int):
        # Выполняем код и складываем результат обратно в messages.
        log.info("Шаг %s: запуск run_python (%s символов)", step, len(code))
        exec_result = execute_python(code, session_dir, data_path)

        if exec_result["figure"]:
            img = exec_result["figure"]
            img["title"] = f"График {len(images) + 1}"
            images.append(img)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": exec_result["text"],
            }
        )

    for step in range(1, MAX_AGENT_STEPS + 1):
        iterations = step
        is_last_step = step >= MAX_AGENT_STEPS
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[RUN_PYTHON_TOOL],
                tool_choice="none" if is_last_step else "auto",
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as e:
            log.exception("Ошибка вызова OpenRouter")
            report = f"Ошибка обращения к LLM: {e}"
            break

        msg = response.choices[0].message
        content = _msg_content(msg)

        # --- Ветка 1: нативный tool call от OpenRouter/OpenAI ---
        if msg.tool_calls:
            # Сохраняем ответ ассистента как dict, чтобы не смешивать типы
            assistant_msg = {"role": "assistant", "content": content}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            messages.append(assistant_msg)

            for tc in msg.tool_calls:
                if tc.function.name != "run_python":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Неизвестный инструмент: {tc.function.name}",
                        }
                    )
                    continue

                try:
                    args = json.loads(tc.function.arguments or "{}")
                    code = args.get("code", "")
                except Exception as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Не удалось разобрать аргументы: {e}",
                        }
                    )
                    continue

                _execute_and_append(tc.id, code, step)
            continue

        # --- Ветка 2: модель без native tool calling, вызов пришёл текстом ---
        parsed = parse_text_tool_call(content)
        if parsed is not None:
            _, code = parsed
            synthetic_id = f"textcall_{uuid.uuid4().hex}"

            # Кладём в историю ответ ассистента с «искусственным» tool_call,
            # чтобы дальнейшее сообщение с role=tool имело валидный tool_call_id.
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": synthetic_id,
                            "type": "function",
                            "function": {
                                "name": "run_python",
                                "arguments": json.dumps(
                                    {"code": code}, ensure_ascii=False
                                ),
                            },
                        }
                    ],
                }
            )
            _execute_and_append(synthetic_id, code, step)
            continue

        # --- Ветка 3: финал. Но отдаём только если это похоже на отчёт ---
        clean = content.strip()
        looks_like_report = (
            len(clean) > 200
            and (
                "ОБЗОР ДАННЫХ" in clean.upper()
                or "КЛЮЧЕВЫЕ МЕТРИКИ" in clean.upper()
                or "ТРЕНДЫ" in clean.upper()
                or "БИЗНЕС-ВЫВОДЫ" in clean.upper()
            )
        )
        if looks_like_report:
            report = clean
            break

        # Иначе модель прислала мусор (например, XML-теги без обёртки
        # <tool_call> или короткий комментарий) — просим продолжить.
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Твой предыдущий ответ не похож ни на вызов run_python, "
                    "ни на финальный отчёт. Если данные ещё не исследованы, "
                    "вызови run_python (нативно или через <tool_call>). Если "
                    "всё уже посчитано, напиши финальный отчёт с разделами: "
                    "Обзор данных, Ключевые метрики, Тренды и закономерности, "
                    "Аномалии, Бизнес-выводы, Ограничения."
                ),
            }
        )

    if not report:
        report = (
            "Агент не успел сформировать отчёт. "
            "Попробуйте увеличить лимит шагов или упростить запрос."
        )

    return {
        "report": report,
        "images": images,
        "iterations": iterations,
    }