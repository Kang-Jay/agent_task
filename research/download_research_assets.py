from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PAPERS = [
    {"id": "embodied_reasoner", "title": "Embodied-Reasoner", "url": "https://arxiv.org/pdf/2503.21696"},
    {"id": "react", "title": "ReAct", "url": "https://arxiv.org/pdf/2210.03629"},
    {"id": "saycan", "title": "SayCan", "url": "https://arxiv.org/pdf/2204.01691"},
    {"id": "voyager", "title": "Voyager", "url": "https://arxiv.org/pdf/2305.16291"},
    {"id": "reflexion", "title": "Reflexion", "url": "https://arxiv.org/pdf/2303.11366"},
    {"id": "sima", "title": "SIMA", "url": "https://arxiv.org/pdf/2404.10179"},
    {"id": "vima", "title": "VIMA", "url": "https://arxiv.org/pdf/2210.03094"},
    {"id": "palm_e", "title": "PaLM-E", "url": "https://arxiv.org/pdf/2303.03378"},
    {"id": "rt_2", "title": "RT-2", "url": "https://arxiv.org/pdf/2307.15818"},
    {"id": "llm_planner", "title": "LLM-Planner", "url": "https://arxiv.org/pdf/2212.04088"},
    {"id": "hamt", "title": "HAMT", "url": "https://arxiv.org/pdf/2110.13309"},
]


CODEBASES = [
    {"id": "ai2thor", "repo": "allenai/ai2thor"},
    {"id": "habitat_lab", "repo": "facebookresearch/habitat-lab"},
    {"id": "langgraph", "repo": "langchain-ai/langgraph"},
    {"id": "mem0", "repo": "mem0ai/mem0"},
    {"id": "llama_index", "repo": "run-llama/llama_index"},
    {"id": "chroma", "repo": "chroma-core/chroma"},
    {"id": "chainlit", "repo": "Chainlit/chainlit"},
]


def download(url: str, path: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "embodied-visual-search-agent"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        if data.strip() in {b"404: Not Found", b"Not Found"}:
            raise urllib.error.URLError("not found payload")
        path.write_bytes(data)
        return {"status": "downloaded", "path": str(path.relative_to(ROOT)), "bytes": str(len(data))}
    except (urllib.error.URLError, TimeoutError) as exc:
        marker = path.with_suffix(path.suffix + ".missing.txt")
        marker.write_text(f"Failed to download {url}\nReason: {exc}\n", encoding="utf-8")
        return {"status": "missing", "path": str(marker.relative_to(ROOT)), "reason": str(exc)}


def download_first_available(urls: list[str], path: Path) -> dict[str, str]:
    last_result: dict[str, str] | None = None
    for url in urls:
        result = download(url, path)
        if result["status"] == "downloaded":
            result["url"] = url
            return result
        last_result = result
    return last_result or {"status": "missing", "path": str(path.relative_to(ROOT)), "reason": "no urls provided"}


def main() -> None:
    paper_results = []
    for paper in PAPERS:
        out = ROOT / "research" / "papers" / f"{paper['id']}.pdf"
        result = download(paper["url"], out)
        paper_results.append({**paper, **result})

    codebase_results = []
    for item in CODEBASES:
        owner_repo = item["repo"]
        readme_path = ROOT / "research" / "codebases" / item["id"] / "README.md"
        license_path = ROOT / "research" / "codebases" / item["id"] / "LICENSE"
        readme = download_first_available(
            [
                f"https://raw.githubusercontent.com/{owner_repo}/main/README.md",
                f"https://raw.githubusercontent.com/{owner_repo}/master/README.md",
                f"https://raw.githubusercontent.com/{owner_repo}/main/readme.md",
                f"https://raw.githubusercontent.com/{owner_repo}/master/readme.md",
            ],
            readme_path,
        )
        license_result = download_first_available(
            [
                f"https://raw.githubusercontent.com/{owner_repo}/main/LICENSE",
                f"https://raw.githubusercontent.com/{owner_repo}/master/LICENSE",
                f"https://raw.githubusercontent.com/{owner_repo}/main/LICENSE.md",
                f"https://raw.githubusercontent.com/{owner_repo}/master/LICENSE.md",
            ],
            license_path,
        )
        codebase_results.append({**item, "readme": readme, "license": license_result})

    forum_notes = ROOT / "research" / "forum" / "community_reception_notes.md"
    forum_notes.parent.mkdir(parents=True, exist_ok=True)
    forum_notes.write_text(
        "# Community Reception Notes\n\n"
        "- ReAct, Voyager, Reflexion, SayCan, PaLM-E, RT-2, SIMA, VIMA, HAMT, DUET, and ObjectNav are repeatedly used as reference points in public embodied-agent discussions.\n"
        "- Zhihu and forum discussions commonly praise ReAct for simple reproducible agent loops, Voyager for skill-memory style self-improvement, and SayCan for grounding actions in feasibility.\n"
        "- This project uses those ideas as architecture references, not as copied source code.\n",
        encoding="utf-8",
    )

    index = {
        "papers": paper_results,
        "codebases": codebase_results,
        "forum_notes": str(forum_notes.relative_to(ROOT)),
        "usage": "Downloaded files are research memory/reference assets. Runtime code remains dependency-light and config-driven.",
    }
    out = ROOT / "research" / "references" / "research_asset_index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
