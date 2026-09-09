"""命令行入口：用于本地快速检验 guide_agent。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from guide_agent.answering import TrustedAnswer
from guide_agent.runtime import (
    RuntimeBuildError,
    build_default_chat_service,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run guide_agent chat in terminal.",
    )
    parser.add_argument(
        "--scene",
        help="场景文件路径。可传相对或绝对路径。",
    )
    parser.add_argument(
        "--question",
        help=(
            "一次性提问；省略则进入交互模式。"
            "输入 exit/quit 退出。"
        ),
    )
    return parser


def _serialize_answer(answer: TrustedAnswer) -> dict[str, Any]:
    return {
        "status": answer.status,
        "answer": answer.answer,
        "tools": list(answer.tools),
        "sources": [
            {
                "source": source.source,
                "chunk_id": source.chunk_id,
                "score": source.score,
            }
            for source in answer.sources
        ],
    }


def _service_init_message() -> None:
    print(
        json.dumps(
            {
                "status": "service_unavailable",
                "answer": (
                    "导览服务初始化失败，请先检查模型参数和场景配置。"
                ),
                "tools": [],
                "sources": [],
            },
            ensure_ascii=False,
        )
    )


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False))


async def _ask_once(
    service,
    question: str,
) -> dict[str, object]:
    try:
        async_answer = getattr(service, "aanswer", None)
        answer = await async_answer(question) if callable(async_answer) else await asyncio.to_thread(service.answer, question)
    except RuntimeBuildError:
        return {
            "status": "service_unavailable",
            "answer": (
                "导览服务在处理问题时异常，请稍后再试。"
            ),
            "tools": [],
            "sources": [],
        }
    except Exception:
        return {
            "status": "internal_error",
            "answer": "服务异常，请稍后再试。",
            "tools": [],
            "sources": [],
        }
    return _serialize_answer(answer)


async def _loop(service) -> int:
    while True:
        try:
            question = input("请输入问题（输入 exit 或 quit 结束）：").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        if len(question) > 1000:
            print("问题最长1000字符，请缩短后再试。")
            continue
        _print_json(await _ask_once(service, question))
    return 0


async def _main_async(args) -> int:
    scene_path = Path(args.scene) if args.scene else None
    try:
        service = build_default_chat_service(scene_path)
    except RuntimeBuildError:
        _service_init_message()
        return 1
    except Exception:
        _print_json({"status": "runtime_error", "answer": "初始化失败。", "tools": [], "sources": []})
        return 1
    try:
        if args.question is None:
            return await _loop(service)
        payload = await _ask_once(service, args.question)
        _print_json(payload)
        return 1 if payload["status"] in {"model_error", "service_unavailable", "internal_error", "empty_response", "max_steps_exceeded"} else 0
    finally:
        close = getattr(service, "aclose", None)
        if callable(close):
            await close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = _build_parser()
    args = parser.parse_args()
    if args.question is not None:
        args.question = args.question.strip()
        if not args.question or len(args.question) > 1000:
            parser.error("问题必须为1到1000个字符，不能只有空白。")

    code = asyncio.run(_main_async(args))
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
