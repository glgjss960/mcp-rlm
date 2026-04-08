from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


_DEFAULT_0SHOT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)".
"""


_DEFAULT_0SHOT_COT = """Please read the following text and answer the questions below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step:
"""


@dataclass
class LongBenchPromptSet:
    zero_shot: str
    zero_shot_cot: str


def _read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\ufeff", "")
    text = text.strip()
    return text or None


def load_prompt_set(prompt_dir: str | None = None) -> LongBenchPromptSet:
    if prompt_dir:
        root = Path(prompt_dir).expanduser().resolve()
        zero = _read_text_if_exists(root / "0shot.txt")
        cot = _read_text_if_exists(root / "0shot_cot.txt")
        return LongBenchPromptSet(
            zero_shot=zero or _DEFAULT_0SHOT,
            zero_shot_cot=cot or _DEFAULT_0SHOT_COT,
        )
    return LongBenchPromptSet(
        zero_shot=_DEFAULT_0SHOT,
        zero_shot_cot=_DEFAULT_0SHOT_COT,
    )


def _safe_choice(choices: Dict[str, str], letter: str) -> str:
    return str(choices.get(letter, "")).strip()


def _fill_template(template: str, *, context: str, question: str, choices: Dict[str, str]) -> str:
    out = str(template)
    out = out.replace("$DOC$", str(context))
    out = out.replace("$Q$", str(question))
    out = out.replace("$C_A$", _safe_choice(choices, "A"))
    out = out.replace("$C_B$", _safe_choice(choices, "B"))
    out = out.replace("$C_C$", _safe_choice(choices, "C"))
    out = out.replace("$C_D$", _safe_choice(choices, "D"))
    return out


def render_longbench_prompt(
    *,
    style: str,
    question: str,
    choices: Dict[str, str],
    context: str,
    prompt_set: LongBenchPromptSet,
) -> str:
    normalized = str(style or "hybrid").strip().lower()
    if normalized == "0shot":
        return _fill_template(prompt_set.zero_shot, context=context, question=question, choices=choices)
    if normalized == "0shot_cot":
        return _fill_template(prompt_set.zero_shot_cot, context=context, question=question, choices=choices)
    if normalized == "internal":
        lines = [
            "Answer the multiple-choice question.",
            "Use only evidence from the provided context.",
            'Output format must be: "The correct answer is (X)".',
            "",
            f"Question: {question}",
            f"(A) {_safe_choice(choices, 'A')}",
            f"(B) {_safe_choice(choices, 'B')}",
            f"(C) {_safe_choice(choices, 'C')}",
            f"(D) {_safe_choice(choices, 'D')}",
            "",
            "Context:",
            context,
        ]
        return "\n".join(lines)

    # hybrid
    official = _fill_template(prompt_set.zero_shot, context=context, question=question, choices=choices)
    prefix = (
        "Follow strict format constraints from LongBench. "
        'Final line must be exactly: "The correct answer is (X)".\n\n'
    )
    return prefix + official

