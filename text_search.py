import json
from typing import Iterable
from urllib import error, request

from config import load_config


class TextSearch:
    def __init__(
        self,
        api_key: str,
        search_url: str,
        max_results: int = 10,
    ) -> None:
        config = load_config()
        self.api_key = api_key or config.serper_api_key
        self.search_url = search_url or config.serper_search_url
        self.max_results = max_results

    def search(self, query: str | Iterable[str]) -> str:
        queries = [query] if isinstance(query, str) else list(query)
        results = [self._search_single(item) for item in queries]
        return "\n=======\n".join(results)

    def _search_single(self, query: str) -> str:
        if not self.api_key:
            return (
                "[Search - API Key Required]\n\n"
                "Set `SERPER_API_KEY` (or `SERP_API_KEY`) to enable web search."
            )

        payload = json.dumps({"q": query, "hl": "en", "gl": "us"}).encode("utf-8")
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        req = request.Request(
            self.search_url,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return f"Search returned HTTP {exc.code} for '{query}'\n{body}"
        except Exception as exc:  # noqa: BLE001
            return f"Search request failed for '{query}': {exc}"

        items = data.get("organic") or []
        snippets: list[str] = []
        for idx, item in enumerate(items[: self.max_results], 1):
            title = item.get("title", "Untitled")
            link = item.get("link", "")
            snippet = (item.get("snippet") or "").strip()
            date = item.get("date")

            entry = f"{idx}. [{title}]({link})"
            if date:
                entry += f"\n   Date published: {date}"
            if snippet:
                entry += f"\n   {snippet}"
            snippets.append(entry)

        if not snippets:
            return f"No search results found for '{query}'"

        return f"Search for '{query}' returned {len(snippets)} results:\n\n" + "\n\n".join(snippets)


def text_search(query: str | Iterable[str]) -> str:
    return TextSearch().search(query)
