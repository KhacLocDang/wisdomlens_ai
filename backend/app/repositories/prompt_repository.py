from __future__ import annotations

from typing import TYPE_CHECKING

from app.models.prompt import Prompt
from app.services.prompt_service import DEFAULT_PROMPTS

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def list_prompts(db: Session) -> list[Prompt]:
    """Return all prompt rows stored in database."""
    return db.query(Prompt).all()


def get_prompt(db: Session, key: str) -> Prompt | None:
    """Retrieve a single prompt record by its stable key."""
    return db.query(Prompt).filter(Prompt.key == key).first()


def save_or_update_prompt(
    db: Session,
    key: str,
    custom_prompt: str | None,
) -> Prompt:
    """Save or update custom prompt text for a given key."""
    if key not in DEFAULT_PROMPTS:
        raise ValueError(f"Unsupported prompt key: {key}")

    meta = DEFAULT_PROMPTS[key]
    record = get_prompt(db, key)

    cleaned_custom = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None

    if record is None:
        record = Prompt(
            key=key,
            name=meta["name"],
            category=meta["category"],
            description=meta["description"],
            custom_prompt=cleaned_custom,
        )
        db.add(record)
    else:
        record.custom_prompt = cleaned_custom

    db.commit()
    db.refresh(record)
    return record


def restore_prompt(db: Session, key: str) -> Prompt | None:
    """Restore a prompt to shipped default by setting custom_prompt to None."""
    if key not in DEFAULT_PROMPTS:
        raise ValueError(f"Unsupported prompt key: {key}")

    record = get_prompt(db, key)
    if record is not None:
        record.custom_prompt = None
        db.commit()
        db.refresh(record)
    return record
