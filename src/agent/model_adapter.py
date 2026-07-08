from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
API_KEY_PATH = ROOT / "apikey.txt"


@dataclass(frozen=True)
class ApiCredential:
    provider: str
    api_key: str
    base_url: str | None = None
    model: str | None = None


def load_credentials(path: Path = API_KEY_PATH) -> list[ApiCredential]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    credentials: list[ApiCredential] = []
    pending_label: str | None = None
    for line in lines:
        if line.endswith(":") and not line.lower().startswith("sk-"):
            pending_label = line[:-1].strip().lower()
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            provider = name.strip().lower()
            api_key = value.strip()
        elif ":" in line and not line.lower().startswith("sk-"):
            name, value = line.split(":", 1)
            provider = name.strip().lower()
            api_key = value.strip()
        else:
            provider = (pending_label or "openai").lower()
            api_key = line
        if api_key.lower().startswith("sk-"):
            credentials.append(_credential_for(provider, api_key))
            pending_label = None
    return credentials


def _credential_for(provider: str, api_key: str) -> ApiCredential:
    normalized = provider.replace(" ", "").replace("_", "").lower()
    if "deepseek" in normalized:
        return ApiCredential(provider="deepseek", api_key=api_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    return ApiCredential(provider="openai_compatible", api_key=api_key, model="gpt-4o-mini")


class ModelAdapter:
    def __init__(self, credentials: list[ApiCredential] | None = None):
        self.credentials = credentials if credentials is not None else load_credentials()

    def available(self) -> bool:
        return bool(self.credentials)

    def audit(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "providers": [
                {
                    "provider": credential.provider,
                    "base_url": credential.base_url or "default",
                    "model": credential.model,
                    "key_length": len(credential.api_key),
                }
                for credential in self.credentials
            ],
        }

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        errors: list[str] = []
        for credential in self.credentials:
            try:
                client = OpenAI(api_key=credential.api_key, base_url=credential.base_url)
                response = client.chat.completions.create(
                    model=credential.model or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1,
                    max_tokens=220,
                    response_format={"type": "json_object"} if credential.provider != "deepseek" else None,
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:
                errors.append(f"{credential.provider}: {type(exc).__name__}: {str(exc)[:160]}")
        return {"error": "all_model_calls_failed", "errors": errors}


def smoke_test() -> dict[str, Any]:
    adapter = ModelAdapter()
    if not adapter.available():
        return {"ok": False, "reason": "no credentials"}
    result = adapter.complete_json(
        "Return only compact JSON.",
        'Return {"ok": true, "action": "TURN_RIGHT"} and no extra text.',
    )
    return {"ok": bool(result.get("ok") or result.get("action")), "result": result, "audit": adapter.audit()}


if __name__ == "__main__":
    print(json.dumps(smoke_test(), ensure_ascii=False, indent=2))

