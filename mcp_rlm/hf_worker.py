from __future__ import annotations

from typing import Any, Dict, List
import argparse
import json
import sys


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _chat_prompt_from_messages(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if tokenizer is not None:
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except TypeError:
                try:
                    rendered = apply_template(messages, tokenize=False)
                    if isinstance(rendered, str) and rendered.strip():
                        return rendered
                except Exception:
                    pass
            except Exception:
                pass
    return (
        "System:\n"
        + str(messages[0].get("content", ""))
        + "\n\nUser:\n"
        + str(messages[1].get("content", ""))
        + "\n\nAssistant:\n"
    )


def _extract_generation_text(first: Any, *, prompt: str) -> str:
    if isinstance(first, dict):
        generated = first.get("generated_text", "")
        if isinstance(generated, str):
            text = generated
        elif isinstance(generated, list):
            candidate = ""
            for item in generated:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip().lower()
                content = str(item.get("content", ""))
                if role == "assistant" and content.strip():
                    candidate = content
            text = candidate or str(generated)
        else:
            text = str(generated)
        if not text and first.get("text") is not None:
            text = str(first.get("text", ""))
    else:
        text = str(first)
    if text.startswith(prompt):
        return text[len(prompt) :]
    return text


def _load_pipeline(args: argparse.Namespace) -> Any:
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("transformers package is required for huggingface policy mode") from exc

    kwargs: Dict[str, Any] = {}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.torch_dtype and args.torch_dtype != "auto":
        kwargs["torch_dtype"] = args.torch_dtype

    try:
        return pipeline("text-generation", model=args.model, tokenizer=args.model, **kwargs)
    except TypeError:
        kwargs.pop("device_map", None)
        kwargs.pop("torch_dtype", None)
        return pipeline("text-generation", model=args.model, tokenizer=args.model, **kwargs)


def _handle_chat(req: Dict[str, Any], text_pipeline: Any) -> Dict[str, Any]:
    req_id = str(req.get("id", ""))
    system = str(req.get("system", ""))
    user = str(req.get("user", ""))
    max_new_tokens = max(8, int(req.get("max_new_tokens", 256)))

    messages = [
        {"role": "system", "content": system + " Return ONLY valid JSON object."},
        {"role": "user", "content": user},
    ]
    prompt = _chat_prompt_from_messages(getattr(text_pipeline, "tokenizer", None), messages)
    try:
        outputs = text_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )
    except TypeError:
        outputs = text_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    if not outputs:
        raise RuntimeError("Empty generation output")
    first = outputs[0]
    text = _extract_generation_text(first, prompt=prompt)
    return {"id": req_id, "ok": True, "text": text}


def main() -> int:
    parser = argparse.ArgumentParser(description="HF worker for MCP-RLM local policy")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    args = parser.parse_args()

    try:
        text_pipeline = _load_pipeline(args)
    except Exception as exc:
        _emit({"ok": False, "error": str(exc), "error_type": type(exc).__name__})
        return 1

    _emit({"ok": True, "type": "ready"})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        req_id = ""
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be JSON object")
            req_id = str(req.get("id", ""))
        except Exception as exc:
            _emit({"id": req_id, "ok": False, "error": str(exc), "error_type": type(exc).__name__})
            continue

        req_type = str(req.get("type", "")).strip().lower()
        if req_type == "shutdown":
            _emit({"id": req_id, "ok": True, "type": "shutdown"})
            return 0
        if req_type != "chat_json":
            _emit({"id": req_id, "ok": False, "error": f"unsupported request type: {req_type}"})
            continue

        try:
            resp = _handle_chat(req, text_pipeline=text_pipeline)
        except Exception as exc:
            resp = {"id": req_id, "ok": False, "error": str(exc), "error_type": type(exc).__name__}
        _emit(resp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
