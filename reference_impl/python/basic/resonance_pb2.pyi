from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SemanticEvent(_message.Message):
    __slots__ = ("source_id", "created_at", "embedding", "debug_label")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    DEBUG_LABEL_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    created_at: int
    embedding: _containers.RepeatedScalarFieldContainer[float]
    debug_label: str
    def __init__(self, source_id: _Optional[str] = ..., created_at: _Optional[int] = ..., embedding: _Optional[_Iterable[float]] = ..., debug_label: _Optional[str] = ...) -> None: ...
