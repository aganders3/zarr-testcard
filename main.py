import uuid
from contextlib import contextmanager
from textwrap import dedent

from typing import TypeAlias, Literal
from typing_extensions import Self

import numcodecs
import numpy as np
import pydantic
from fastapi import FastAPI, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import (
    create_engine,
    select,
    Column,
    Integer,
    String,
    JSON,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

BLOSC_ENCODER = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)

engine = create_engine("sqlite+pysqlite:///zarr-testcard.sqlite", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session():
    with SessionLocal() as session:
        try:
            yield session
            session.commit()
        except Exception:
            # TODO: log the exception, but don't expose it
            session.rollback()
            raise


session_context = contextmanager(get_session)


def set_up_db():
    Base.metadata.create_all(bind=engine)


class ZarrRecord(Base):
    __tablename__ = "zarrs"

    id = Column(Integer, primary_key=True)
    external_id = Column(String, unique=True, nullable=False)
    axes = Column(JSON, nullable=False)
    shape = Column(JSON, nullable=False)
    chunks = Column(JSON, nullable=False)
    scales = Column(JSON, nullable=False)

    def get_by_external_id(db: Session, external_id: str) -> Self:
        return db.execute(
            select(ZarrRecord).where(ZarrRecord.external_id == external_id)
        ).scalar()


set_up_db()


ZarrID: TypeAlias = str
AxisName: TypeAlias = Literal["T", "C", "Z", "Y", "X"]


class Axis(pydantic.BaseModel):
    name: AxisName
    # TODO: unit and type validation - can zarr-python do this?
    axis_type: str = pydantic.Field(alias="type")
    unit: str | None = None


class ZarrCreate(pydantic.BaseModel):
    axes: list[Axis]
    shape: list[int]
    chunks: list[list[int]]
    scales: list[list[float]] | None = None

    @pydantic.model_validator(mode="after")
    def validate(self) -> Self:
        num_axes = len(self.axes)

        if len(self.shape) != num_axes:
            raise ValueError("shape must have the same length as axes")

        if any(len(chunks) != num_axes for chunks in self.chunks):
            raise ValueError("chunks must have the same length as axes")

        # TODO: validate chunk shapes?

        if self.scales is not None and any(
            len(scale) != num_axes for scale in self.scales
        ):
            raise ValueError("scales must be None or have the same length as axes")
        return self

    def model_dump(self, **kwargs):
        kwargs.setdefault("by_alias", True)
        kwargs.setdefault("exclude_defaults", True)
        return super().model_dump(**kwargs)


class Zarr(ZarrCreate):
    zarr_id: ZarrID

    @staticmethod
    def from_record(record: ZarrRecord) -> Self:
        return Zarr(
            zarr_id=record.external_id,
            axes=[Axis(**x) for x in record.axes],
            shape=record.shape,
            chunks=record.chunks,
            scales=record.scales,
        )


@app.post("/")
def new_zarr(zarr_data: ZarrCreate, db: Session = Depends(get_session)) -> Zarr:
    zarr_id = _get_zarr_id()
    db.add(
        ZarrRecord(
            external_id=zarr_id,
            **zarr_data.model_dump(),
        )
    )
    db.commit()
    record = ZarrRecord.get_by_external_id(db, zarr_id)
    return Zarr.from_record(record)


@app.get("/{zarr_id}")
def get_zarr(zarr_id: ZarrID, db: Session = Depends(get_session)):
    print("GET", zarr_id)
    if record := ZarrRecord.get_by_external_id(db, zarr_id):
        return Zarr.from_record(record)
    return Response(status_code=404)


@app.get("/{zarr_id}/.zgroup")
def get_zgroup(zarr_id: ZarrID):
    return {"zarr_format": 2}


@app.get("/{zarr_id}/.zattrs")
def get_zattrs(zarr_id: ZarrID, db: Session = Depends(get_session)):
    record = ZarrRecord.get_by_external_id(db, zarr_id)
    return {
        "multiscales": [
            {
                "axes": record.axes,
                "coordinateTransformations": [{"type": "identity"}],
                "datasets": [
                    {
                        "coordinateTransformations": [{"scale": s, "type": "scale"}],
                        "path": str(i),
                    }
                    for i, s in enumerate(record.scales)
                ],
                "name": zarr_id,
                "version": "0.4",
            }
        ],
    }


@app.get("/{zarr_id}/{scale}/.zarray")
def get_zarray(
    zarr_id: ZarrID,
    scale: int,
    db: Session = Depends(get_session),
):
    record = ZarrRecord.get_by_external_id(db, zarr_id)
    return {
        "chunks": record.chunks[scale],
        "compressor": {
            "blocksize": 0,
            "clevel": 1,
            "cname": "zstd",
            "id": "blosc",
            "shuffle": 2,
        },
        "dimension_separator": "/",
        # TODO: support other dtypes
        "dtype": "|u1",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": _get_shapes_from_scales(record.shape, record.scales)[scale],
        "zarr_format": 2,
    }


def _get_zarr_id() -> ZarrID:
    # TODO: use a friendly hash
    return str(uuid.uuid4())


def _get_shapes_from_scales(shape, scales):
    # shape of the array is relative to the first scale factor
    return [
        [int(np.ceil(i / (s / s0))) for i, s, s0 in zip(shape, scale, scales[0])]
        for scale in scales
    ]


def _get_chunk_offsets(t: int, c: int, z: int, y: int, x: int, chunks):
    return tuple(
        axis * chunk_size
        for axis, chunk_size in zip(
            (t, c, z, y, x),
            chunks,
        )
    )


def generate_chunk(
    chunks,
    shape,
    scales,
    scale: int,
    t: int,
    c: int,
    z: int,
    y: int,
    x: int,
):
    offsets = _get_chunk_offsets(t, c, z, y, x, chunks)
    chunk = np.zeros(chunks, dtype=np.uint8)

    # TODO: set font size based on chunk size (try to make it fit)
    font_size = 80 / scales[-2]
    font = ImageFont.load_default(font_size)
    # loop over t, c, and z
    for t_index, timepoint in enumerate(chunk):
        for c_index, channel in enumerate(timepoint):
            for z_index, zslice in enumerate(channel):
                rows, cols = zslice.shape
                Y, X = np.meshgrid(
                    np.linspace(0, 1, rows, endpoint=False),
                    np.linspace(0, 1, cols, endpoint=False),
                    indexing="ij",
                )
                z_factor = (1 + z_index / channel.shape[0]) / 2
                match (offsets[1] + c_index) % 8:
                    case 0:
                        local_gradient = X * Y * z_factor
                    case 1:
                        local_gradient = (1 - X) * Y * z_factor
                    case 2:
                        local_gradient = X * (1 - Y) * z_factor
                    case 3:
                        local_gradient = (1 - X) * (1 - Y) * z_factor
                    case 4:
                        local_gradient = X * Y * (1 - z_factor)
                    case 5:
                        local_gradient = (1 - X) * Y * (1 - z_factor)
                    case 6:
                        local_gradient = X * (1 - Y) * (1 - z_factor)
                    case 7:
                        local_gradient = (1 - X) * (1 - Y) * (1 - z_factor)
                global_x = X * cols + offsets[-1] - shape[-1] / 2
                global_y = Y * rows + offsets[-2] - shape[-2] / 2
                global_z = offsets[-3] + z_index - shape[-3] / 2
                global_radius = np.sqrt(
                    (global_x / shape[-1] / 2) ** 2
                    + (global_y / shape[-2] / 2) ** 2
                    + (global_z / shape[-3] / 2) ** 2
                )
                global_angle = (
                    np.arctan2(global_y, global_x)
                    + (offsets[0] + t_index) * np.pi / shape[0]
                )
                slice = (
                    0.25
                    + 0.5 * local_gradient
                    + 0.25 * np.sin(32 * global_angle) * (global_radius < 0.25)
                )
                slice = (slice * 255).astype(np.uint8)
                slice[global_x > shape[-1] / 2] = 0
                slice[global_y > shape[-2] / 2] = 0
                slice = Image.fromarray(slice, mode="L")
                draw = ImageDraw.Draw(slice)
                draw.text(
                    (0, 0),
                    dedent(f"""\
                    scale={scale},
                    t={t}, c={c},
                    z={z}, y={y}, x={x}
                """),
                    font=font,
                    fill=255,
                )
                draw.text(
                    (0, zslice.shape[0] - 1.2 * font_size),
                    f"chunk={chunk.shape}",
                    font=font,
                    fill=255,
                )
                zslice[...] = np.array(slice)
    return chunk


@app.get("/{zarr_id}/{chunk_loc:path}")
def get_chunk(
    zarr_id: ZarrID,
    chunk_loc: str,
    db: Session = Depends(get_session)
):
    record = ZarrRecord.get_by_external_id(db, zarr_id)
    if record is None:
        return Response(status_code=404)

    chunk_loc = chunk_loc.split("/")
    try:
        scale = int(chunk_loc.pop(0))
    except ValueError:
        return Response(status_code=404)

    shape = _get_shapes_from_scales(record.shape, record.scales)[scale]
    axes = [axis["name"] for axis in record.axes]

    try:
        chunk_locs = {axis: int(loc) for axis, loc in zip(axes, chunk_loc)}
    except ValueError:
        return Response(status_code=404)

    chunk = generate_chunk(
        record.chunks[scale],
        shape,
        record.scales[scale],
        scale,
        chunk_locs.get("T", 0),
        chunk_locs.get("C", 0),
        chunk_locs.get("Z", 0),
        chunk_locs.get("Y", 0),
        chunk_locs.get("X", 0),
    )
    compressed_chunk = BLOSC_ENCODER.encode(chunk)
    return Response(
        content=compressed_chunk,
        media_type="application/octet-stream",
    )


DEFAULT_ZARR = ZarrCreate(
    axes=(
        {"name": "T", "type": "time", "unit": "second"},
        {"name": "C", "type": "channel"},
        {"name": "Z", "type": "distance", "unit": "micrometer"},
        {"name": "Y", "type": "distance", "unit": "micrometer"},
        {"name": "X", "type": "distance", "unit": "micrometer"},
    ),
    shape=(128, 3, 128, 2048, 4096),
    chunks=(
        (1, 1, 16, 512, 1024),
        (1, 1, 16, 256, 512),
        (1, 1, 8, 128, 256),
    ),
    scales=(
        (1.0, 1.0, 10.0, 1.0, 1.0),
        (1.0, 1.0, 10.0, 2.0, 2.0),
        (1.0, 1.0, 20.0, 4.0, 4.0),
    ),
)
try:
    with session_context() as db:
        db.add(
            ZarrRecord(
                external_id="default",
                **DEFAULT_ZARR.model_dump(),
            )
        )
except IntegrityError:
    pass
