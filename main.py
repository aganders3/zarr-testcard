import uuid
from contextlib import contextmanager
from itertools import product

from typing import Literal, Protocol, TypeAlias, runtime_checkable
from typing_extensions import Self

import numcodecs
import numpy as np
import pydantic
import zarr
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

        # TODO: validate chunk shapes? do they need to be divisible into the shape?

        if self.scales is not None and any(
            len(scale) != num_axes for scale in self.scales
        ):
            raise ValueError("scales must be None or have the same length as axes")
        return self

    def model_dump(self, **kwargs):
        kwargs.setdefault("by_alias", True)
        kwargs.setdefault("exclude_defaults", True)
        return super().model_dump(**kwargs)


# TODO: rename ZarrShape?
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
    font_size = chunk.shape[-2] // 6
    font = ImageFont.load_default(font_size)
    # loop over t, c, and z *within the chunk*
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
                draw.text((0, 0), f"s{scale}", font=font, fill=255)
                draw.text((0, 1.2 * font_size), f"t{t}, c{c}", font=font, fill=255)
                draw.text(
                    (0, 2.4 * font_size), f"z{z}, y{y}, x{x}", font=font, fill=255
                )
                draw.text(
                    (0, zslice.shape[0] - 1.2 * font_size),
                    f"{chunk.shape}",
                    font=font,
                    fill=255,
                )
                zslice[...] = np.array(slice)
    return chunk


@app.get("/{zarr_id}/{chunk_loc:path}")
def get_chunk(zarr_id: ZarrID, chunk_loc: str, db: Session = Depends(get_session)):
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


@runtime_checkable
class ChunkGenerator(Protocol):
    shape: list[int]
    chunks: list[int]
    dtype: np.dtype
    num_chunks: list[int]
    def __call__(self, chunk_loc: list[int]) -> np.ndarray: ...


class TestCardGenerator:
    def __init__(self, shape_metadata: Zarr, scale: int = 0, dtype=np.uint8):
        self._shape_metadata = shape_metadata
        self.dtype = dtype
        self._scale = scale

    def __call__(self, chunk_loc: list[int]):
        try:
            t, c, z, y, x = chunk_loc
        except ValueError:
            raise ValueError("chunk_loc must have 5 elements (t, c, z, y, x)")

        chunk = np.zeros(self.chunks, dtype=self.dtype)

        offsets = self._get_chunk_offsets(*chunk_loc)
        c_offset = offsets[1]
        t_offset = offsets[0]

        # loop over t, c, and z *within the chunk*
        for local_t, timepoint in enumerate(chunk):
            for local_c, channel in enumerate(timepoint):
                for local_z, zslice in enumerate(channel):
                    local_y, local_x = self._get_local_coords(zslice.shape)
                    global_x, global_y, global_z = self._get_global_coords(
                        local_x, local_y, local_z, offsets
                    )
                    # linear gradient provides local context + channel variation
                    local_gradient = self._get_local_gradient(
                        local_x,
                        local_y,
                        (1 + local_z / channel.shape[0]) / 2,
                        c_offset + local_c,
                    )
                    # radial gradient provides global context + time variation
                    radial_gradient = self._get_radial_gradient(
                        global_x, global_y, global_z, t_offset + local_t
                    )
                    slice = 0.5 * local_gradient + 0.5 * radial_gradient
                    slice = (slice * 255).astype(self.dtype)
                    # slice is annotated with chunk location + chunk size
                    slice = self._annotate_slice(slice, chunk_loc)
                    zslice[...] = slice
        return chunk

    def __len__(self) -> int:
        return np.prod([s // c for s, c in zip(self.shape, self.chunks)])

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def shape(self):
        total_shape = self._shape_metadata.shape
        scale0 = self._shape_metadata.scales[0]
        scaleN = self._shape_metadata.scales[self.scale]
        return [
            int(np.ceil(i / (s / s0))) for i, s, s0 in zip(total_shape, scaleN, scale0)
        ]

    @property
    def chunks(self):
        return self._shape_metadata.chunks[self._scale]

    @property
    def scales(self):
        return self._shape_metadata.scales[self._scale]

    @property
    def num_chunks(self):
        return [int(np.ceil(s / c)) for s, c in zip(self.shape, self.chunks)]

    def _get_chunk_offsets(self, t: int, c: int, z: int, y: int, x: int):
        return tuple(
            axis * chunk_size
            for axis, chunk_size in zip(
                (t, c, z, y, x),
                self.chunks,
            )
        )

    def _get_local_coords(self, slice_shape):
        rows, cols = slice_shape
        Y, X = np.meshgrid(
            np.linspace(0, 1, rows, endpoint=False),
            np.linspace(0, 1, cols, endpoint=False),
            indexing="ij",
        )
        return Y, X

    def _get_global_coords(self, local_x, local_y, local_z, offsets):
        _, _, z_offset, y_offset, x_offset = offsets
        _, _, z_shape, y_shape, x_shape = self.shape
        _, _, _, y_chunk_size, x_chunk_size = self.chunks
        global_x = local_x * x_chunk_size + x_offset - x_shape / 2
        global_y = local_y * y_chunk_size + y_offset - y_shape / 2
        global_z = local_z + z_offset - z_shape / 2
        return global_x, global_y, global_z

    def _get_local_gradient(self, X, Y, z_factor, channel):
        match (channel) % 8:
            case 0:
                return X * Y * z_factor
            case 1:
                return (1 - X) * Y * z_factor
            case 2:
                return X * (1 - Y) * z_factor
            case 3:
                return (1 - X) * (1 - Y) * z_factor
            case 4:
                return X * Y * (1 - z_factor)
            case 5:
                return (1 - X) * Y * (1 - z_factor)
            case 6:
                return X * (1 - Y) * (1 - z_factor)
            case 7:
                return (1 - X) * (1 - Y) * (1 - z_factor)

    def _get_radial_gradient(self, global_x, global_y, global_z, t):
        t_shape, _, z_shape, y_shape, x_shape = self.shape
        global_radius = np.sqrt(
            (global_x / x_shape / 2) ** 2
            + (global_y / y_shape / 2) ** 2
            + (global_z / z_shape / 2) ** 2
        )
        global_angle = np.arctan2(global_y, global_x) + t * np.pi / t_shape
        return 0.5 * (1 + np.sin(32 * global_angle)) * (global_radius < 0.25)

    def _annotate_slice(self, slice: np.ndarray, chunk_loc: list[int]):
        t, c, z, y, x = chunk_loc
        font_size = slice.shape[-2] // 6
        font = ImageFont.load_default(font_size)
        slice_im = Image.fromarray(slice, mode="L")
        draw = ImageDraw.Draw(slice_im)
        draw.text((0, 0), f"s{self.scale}", font=font, fill=255)
        draw.text((0, 1.2 * font_size), f"t{t}, c{c}", font=font, fill=255)
        draw.text((0, 2.4 * font_size), f"z{z}, y{y}, x{x}", font=font, fill=255)
        draw.text(
            (0, slice.shape[0] - 1.2 * font_size),
            f"{self.chunks}",
            font=font,
            fill=255,
        )
        return np.asarray(slice_im)


class ComputeStore:

    def __init__(
        self,
        chunk_computer: ChunkGenerator,
        compressor=BLOSC_ENCODER,
        dimension_separator: Literal[".", "/"] = ".",
    ):
        self._chunk_computer = chunk_computer
        self.compressor = compressor
        self.dimension_separator = dimension_separator

    def getsize(self):
        return 0

    def listdir(self):
        return [".zarray"]

    def __getitem__(self, key):
        if key not in self:
            raise KeyError

        if key == ".zarray":
            return self._zarray

        try:
            return self._get_chunk(self._chunk_loc_for_key(key))
        except ValueError:
            raise KeyError

    def _get_chunk(self, chunk_loc: list[int]):
        return self.compressor.encode(self._chunk_computer(chunk_loc))

    def __delitem__(self, key):
        raise zarr.errors.ReadOnlyError

    def __setitem__(self, key, value):
        raise zarr.errors.ReadOnlyError

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return key == ".zarray" or self._is_valid_chunk_loc(key)

    def _is_valid_chunk_loc(self, key: str) -> bool:
        try:
            chunk_loc = self._chunk_loc_for_key(key)
            return len(chunk_loc) == len(self._chunk_computer.shape) and all(
                0 <= p < s for p, s in zip(chunk_loc, self._chunk_computer.num_chunks)
            )
        except ValueError:
            return False

    def _chunk_loc_for_key(self, key: str) -> list[int]:
        return [int(p) for p in key.split(self.dimension_separator)]

    def keys(self):
        yield ".zarray"
        yield from (
            self.dimension_separator.join([str(p) for p in chunk_loc])
            for chunk_loc in product(*(range(n) for n in self._chunk_computer.num_chunks))
        )

    def __iter__(self):
        return self.keys()

    def values(self):
        return (self[key] for key in self.keys())

    @property
    def _zarray(self):
        return {
            "chunks": self._chunk_computer.chunks,
            "compressor": self.compressor.get_config(),
            "dimension_separator": self.dimension_separator,
            "dtype": str(np.dtype(self._chunk_computer.dtype)),
            "fill_value": 0,
            "filters": None,
            "order": "C",
            "shape": self._chunk_computer.shape,
            "zarr_format": 2,
        }

    def open_kwargs(self):
        return {
            "store": self,
            "mode": "r",
            "shape": self._chunk_computer.shape,
            "chunks": self._chunk_computer.chunks,
            "dtype": self._chunk_computer.dtype,
            "compressor": self.compressor,
        }


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


generator = TestCardGenerator(DEFAULT_ZARR, scale=2)
test_card_store = ComputeStore(generator)

z0 = zarr.open(**test_card_store.open_kwargs())
