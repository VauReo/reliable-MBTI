from dataclasses import dataclass, field


@dataclass(slots=True)
class UserRecord:
    user_id: str
    posts: list[str] = field(default_factory=list)
    mbti_label: str = ""
