import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    serper_api_key: str | None
    serper_search_url: str


def load_config() -> Config:
    return Config(
        serper_api_key="74e39017c9b78f5bc2b3dde8030a83fb8ad28e87",
        serper_search_url=os.getenv("SERPER_SEARCH_URL", "https://google.serper.dev/search"),
    )
