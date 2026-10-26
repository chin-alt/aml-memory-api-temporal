"""固定的文本 Add/Search 请求合同，不接收调用方内核配置。"""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints, field_validator

Text = Annotated[str, StringConstraints(strict=True, min_length=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def nonblank(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        return value  # 校验但不改写 ID/正文。


class Message(ContractModel):
    role: Literal["user", "assistant"]
    content: Text
    timestamp: StrictInt | None = None

    @field_validator("timestamp")
    @classmethod
    def valid_timestamp(cls, value):
        if value is not None:
            try:
                datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            except (ValueError, OverflowError, OSError) as exc:
                raise ValueError("timestamp must be representable Unix milliseconds") from exc
        return value


class AddRequest(ContractModel):
    request_id: Text
    messages: Annotated[list[Message], Field(min_length=1)]
    user_id: Text
    session_id: Text


class SearchRequest(ContractModel):
    query: Text
    user_id: Text
    top_k: Annotated[StrictInt, Field(ge=1, le=100)]
    options: list[Text] | None = None

    @field_validator("options")
    @classmethod
    def valid_options(cls, value):
        if value is not None and any(not item.strip() for item in value):
            raise ValueError("options must not contain blank strings")
        return value


class AddResponse(ContractModel):
    success: Literal[True]
    request_id: Text
    user_id: Text
    session_id: Text


class MemoryItem(ContractModel):
    id: Text
    content: Text
    score: float | None = None
    created_at: str | None = None


class SearchResponse(ContractModel):
    data: list[MemoryItem]


class EvaluationSeed(ContractModel):
    """仅本地仓库评测模式接受；正文已由原 Dataset 按 dialogue_turn 格式化。"""
    occurred_at: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class EvaluationAddRequest(AddRequest):
    messages: Annotated[list[Message], Field(min_length=1, max_length=1)]
    evaluation_seed: EvaluationSeed


class EvaluationSearchRequest(SearchRequest):
    top_k: Annotated[StrictInt, Field(ge=1, le=200)]
