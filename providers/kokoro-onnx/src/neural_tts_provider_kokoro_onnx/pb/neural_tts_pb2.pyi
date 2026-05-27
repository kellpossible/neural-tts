from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Gender(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    GENDER_UNSPECIFIED: _ClassVar[Gender]
    MALE: _ClassVar[Gender]
    FEMALE: _ClassVar[Gender]
    NEUTRAL: _ClassVar[Gender]
GENDER_UNSPECIFIED: Gender
MALE: Gender
FEMALE: Gender
NEUTRAL: Gender

class Voice(_message.Message):
    __slots__ = ("id", "language", "gender", "display_name")
    ID_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    id: str
    language: str
    gender: Gender
    display_name: str
    def __init__(self, id: _Optional[str] = ..., language: _Optional[str] = ..., gender: _Optional[_Union[Gender, str]] = ..., display_name: _Optional[str] = ...) -> None: ...

class WarmupRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SynthesizeRequest(_message.Message):
    __slots__ = ("voice", "speed", "lang", "text")
    VOICE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    LANG_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    voice: str
    speed: float
    lang: str
    text: str
    def __init__(self, voice: _Optional[str] = ..., speed: _Optional[float] = ..., lang: _Optional[str] = ..., text: _Optional[str] = ...) -> None: ...

class ListVoicesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ShutdownRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RegisterClonedVoiceRequest(_message.Message):
    __slots__ = ("id", "display_name", "language", "reference_audio_wav")
    ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_AUDIO_WAV_FIELD_NUMBER: _ClassVar[int]
    id: str
    display_name: str
    language: str
    reference_audio_wav: bytes
    def __init__(self, id: _Optional[str] = ..., display_name: _Optional[str] = ..., language: _Optional[str] = ..., reference_audio_wav: _Optional[bytes] = ...) -> None: ...

class RemoveClonedVoiceRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class Request(_message.Message):
    __slots__ = ("warmup", "synthesize", "list_voices", "shutdown", "register_cloned_voice", "remove_cloned_voice")
    WARMUP_FIELD_NUMBER: _ClassVar[int]
    SYNTHESIZE_FIELD_NUMBER: _ClassVar[int]
    LIST_VOICES_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    REGISTER_CLONED_VOICE_FIELD_NUMBER: _ClassVar[int]
    REMOVE_CLONED_VOICE_FIELD_NUMBER: _ClassVar[int]
    warmup: WarmupRequest
    synthesize: SynthesizeRequest
    list_voices: ListVoicesRequest
    shutdown: ShutdownRequest
    register_cloned_voice: RegisterClonedVoiceRequest
    remove_cloned_voice: RemoveClonedVoiceRequest
    def __init__(self, warmup: _Optional[_Union[WarmupRequest, _Mapping]] = ..., synthesize: _Optional[_Union[SynthesizeRequest, _Mapping]] = ..., list_voices: _Optional[_Union[ListVoicesRequest, _Mapping]] = ..., shutdown: _Optional[_Union[ShutdownRequest, _Mapping]] = ..., register_cloned_voice: _Optional[_Union[RegisterClonedVoiceRequest, _Mapping]] = ..., remove_cloned_voice: _Optional[_Union[RemoveClonedVoiceRequest, _Mapping]] = ...) -> None: ...

class Error(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class WarmupResponse(_message.Message):
    __slots__ = ("sample_rate", "voices")
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    VOICES_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    voices: _containers.RepeatedCompositeFieldContainer[Voice]
    def __init__(self, sample_rate: _Optional[int] = ..., voices: _Optional[_Iterable[_Union[Voice, _Mapping]]] = ...) -> None: ...

class SynthesizeResponseHeader(_message.Message):
    __slots__ = ("sample_rate",)
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    def __init__(self, sample_rate: _Optional[int] = ...) -> None: ...

class AudioChunk(_message.Message):
    __slots__ = ("pcm",)
    PCM_FIELD_NUMBER: _ClassVar[int]
    pcm: bytes
    def __init__(self, pcm: _Optional[bytes] = ...) -> None: ...

class AudioEnd(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListVoicesResponse(_message.Message):
    __slots__ = ("voices",)
    VOICES_FIELD_NUMBER: _ClassVar[int]
    voices: _containers.RepeatedCompositeFieldContainer[Voice]
    def __init__(self, voices: _Optional[_Iterable[_Union[Voice, _Mapping]]] = ...) -> None: ...

class ShutdownResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RegisterClonedVoiceResponse(_message.Message):
    __slots__ = ("voice",)
    VOICE_FIELD_NUMBER: _ClassVar[int]
    voice: Voice
    def __init__(self, voice: _Optional[_Union[Voice, _Mapping]] = ...) -> None: ...

class RemoveClonedVoiceResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Response(_message.Message):
    __slots__ = ("error", "warmup", "synthesize_header", "audio_chunk", "audio_end", "list_voices", "shutdown", "register_cloned_voice", "remove_cloned_voice")
    ERROR_FIELD_NUMBER: _ClassVar[int]
    WARMUP_FIELD_NUMBER: _ClassVar[int]
    SYNTHESIZE_HEADER_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    AUDIO_END_FIELD_NUMBER: _ClassVar[int]
    LIST_VOICES_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    REGISTER_CLONED_VOICE_FIELD_NUMBER: _ClassVar[int]
    REMOVE_CLONED_VOICE_FIELD_NUMBER: _ClassVar[int]
    error: Error
    warmup: WarmupResponse
    synthesize_header: SynthesizeResponseHeader
    audio_chunk: AudioChunk
    audio_end: AudioEnd
    list_voices: ListVoicesResponse
    shutdown: ShutdownResponse
    register_cloned_voice: RegisterClonedVoiceResponse
    remove_cloned_voice: RemoveClonedVoiceResponse
    def __init__(self, error: _Optional[_Union[Error, _Mapping]] = ..., warmup: _Optional[_Union[WarmupResponse, _Mapping]] = ..., synthesize_header: _Optional[_Union[SynthesizeResponseHeader, _Mapping]] = ..., audio_chunk: _Optional[_Union[AudioChunk, _Mapping]] = ..., audio_end: _Optional[_Union[AudioEnd, _Mapping]] = ..., list_voices: _Optional[_Union[ListVoicesResponse, _Mapping]] = ..., shutdown: _Optional[_Union[ShutdownResponse, _Mapping]] = ..., register_cloned_voice: _Optional[_Union[RegisterClonedVoiceResponse, _Mapping]] = ..., remove_cloned_voice: _Optional[_Union[RemoveClonedVoiceResponse, _Mapping]] = ...) -> None: ...
