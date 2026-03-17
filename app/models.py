from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CharacterProfile(Base):
    __tablename__ = "character_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    trigger_token: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    visual_traits: Mapped[str] = mapped_column(Text, nullable=False)
    default_outfit: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    negative_prompt: Mapped[str] = mapped_column(Text, default="", nullable=False)
    style_tags_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    shot_list_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    base_model_ref: Mapped[str] = mapped_column(String(160), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    generation_jobs: Mapped[list["GenerationJob"]] = relationship(
        back_populates="character",
        cascade="all, delete-orphan",
        order_by="desc(GenerationJob.created_at)",
    )
    dataset_images: Mapped[list["DatasetImage"]] = relationship(
        back_populates="character",
        cascade="all, delete-orphan",
        order_by="desc(DatasetImage.created_at)",
    )
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="character",
        cascade="all, delete-orphan",
        order_by="desc(TrainingRun.created_at)",
    )


class GenerationJob(Base):
    __tablename__ = "generation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("character_profiles.id"), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(24), default="dataset", nullable=False)
    source_shot: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=1024, nullable=False)
    height: Mapped[int] = mapped_column(Integer, default=1024, nullable=False)
    steps: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    cfg: Mapped[float] = mapped_column(Float, default=6.0, nullable=False)
    lora_path: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    lora_weight: Mapped[float] = mapped_column(Float, default=0.8, nullable=False)
    runpod_job_id: Mapped[str] = mapped_column(String(120), default="", nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False, index=True)
    output_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    character: Mapped["CharacterProfile"] = relationship(back_populates="generation_jobs")
    dataset_images: Mapped[list["DatasetImage"]] = relationship(back_populates="generation_job")


class DatasetImage(Base):
    __tablename__ = "dataset_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("character_profiles.id"), nullable=False, index=True)
    generation_job_id: Mapped[int | None] = mapped_column(ForeignKey("generation_jobs.id"), nullable=True, index=True)
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)
    remote_image_path: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    caption_tags_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    keep_status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    split: Mapped[str] = mapped_column(String(24), default="train", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    character: Mapped["CharacterProfile"] = relationship(back_populates="dataset_images")
    generation_job: Mapped["GenerationJob | None"] = relationship(back_populates="dataset_images")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("character_profiles.id"), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    preset_name: Mapped[str] = mapped_column(String(80), nullable=False)
    kept_image_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    runpod_job_id: Mapped[str] = mapped_column(String(120), default="", nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False, index=True)
    output_lora_path: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    sample_previews_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    logs_path: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    config_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )

    character: Mapped["CharacterProfile"] = relationship(back_populates="training_runs")
