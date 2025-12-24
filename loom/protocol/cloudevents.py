"""
CloudEvents v1.0 Implementation for Loom
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class CloudEvent(BaseModel):
    """
    CloudEvents 1.0 Specification Implementation.

    Attributes:
        specversion: The version of the CloudEvents specification which the event uses.
        id: Identifies the event.
        source: Identifies the context in which an event happened.
        type: Describes the type of event related to the originating occurrence.
        datacontenttype: Content type of data value.
        dataschema: Identifies the schema that data adheres to.
        subject: Describes the subject of the event in the context of the event producer (identified by source).
        time: Timestamp of when the occurrence happened.
        data: The event payload.
        traceparent: W3C Trace Context (Extension)
    """

    # Required Attributes
    specversion: str = "1.0"
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    type: str # e.g., "node.call", "agent.thought"

    # Optional Attributes
    datacontenttype: str | None = "application/json"
    dataschema: str | None = None
    subject: str | None = None
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: Any | None = None

    # Extensions
    traceparent: str | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        populate_by_name=True,
        extra='allow'
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to handle datetime serialization."""
        data = super().model_dump(**kwargs)
        if 'time' in data and isinstance(data['time'], datetime):
            data['time'] = data['time'].isoformat()
        return data

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard CloudEvents dictionary structure."""
        return self.model_dump(exclude_none=True, by_alias=True)

    @classmethod
    def create(
        cls,
        source: str,
        type: str,
        data: Any | None = None,
        subject: str | None = None,
        traceparent: str | None = None
    ) -> CloudEvent:
        """Factory method to create a CloudEvent."""
        return cls(
            source=source,
            type=type,
            data=data,
            subject=subject,
            traceparent=traceparent
        )
