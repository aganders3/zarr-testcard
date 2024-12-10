from textwrap import dedent
from functools import lru_cache

import fastapi
import numcodecs
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

# chunk size (t, c, z, y, x)
# IMAGE_SIZE = (300, 3, 1, 2 * 1440, 2 * 1920)
IMAGE_SIZE = (300, 3, 1, 5760, 7680)
# IMAGE_SIZE = (300, 3, 1, 2880, 3840)
CHUNK_SIZE = (1, 1, 1, 720, 960)
SCALES = (
    (1.0, 1.0, 1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0, 2.0, 2.0),
    (1.0, 1.0, 1.0, 4.0, 4.0),
    (1.0, 1.0, 1.0, 8.0, 8.0),
)
BLOSC_ENCODER = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)


def get_chunks_and_shape(scale: int):
    scale = SCALES[scale]
    chunks = tuple(int(np.ceil(i / s)) for i, s in zip(CHUNK_SIZE, scale))
    shape = tuple(int(np.ceil(i / s)) for i, s in zip(IMAGE_SIZE, scale))
    return chunks, shape, scale


def get_chunk_offsets(t: int, c: int, z: int, y: int, x: int, chunks):
    return tuple(
        axis * chunk_size
        for axis, chunk_size in zip(
            (t, c, z, y, x),
            chunks,
        )
    )


@lru_cache(maxsize=None)
def generate_chunk(scale: int, t: int, c: int, z: int, y: int, x: int):
    chunks, shape, scales = get_chunks_and_shape(scale)
    offsets = get_chunk_offsets(t, c, z, y, x, chunks)
    chunk = np.zeros(chunks, dtype=np.uint8)

    font_size = 80 / scales[-2]
    font = ImageFont.load_default(font_size)
    # loop over t, c, and z
    for t, timepoint in enumerate(chunk):
        for ch in timepoint:
            for zslice in ch:
                Y, X = np.mgrid[0:zslice.shape[0], 0:zslice.shape[1]]
                local_gradient = X * Y
                local_gradient = local_gradient / local_gradient.max()
                global_x = X + offsets[-1] - shape[-1] / 2
                global_y = Y + offsets[-2] - shape[-2] / 2
                global_radius = np.sqrt(
                    (global_x / shape[-1] / 2) ** 2 + (global_y / shape[-2] / 2) ** 2
                )
                global_angle = np.arctan2(global_y, global_x) + (offsets[0] + t) * np.pi / shape[0]
                slice = (
                    0.25
                    + 0.5 * local_gradient
                    + 0.25 * np.sin(32 * global_angle) * (global_radius < 0.25)
                )
                slice = (slice * 255).astype(np.uint8)
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


@app.get("/.zattrs")
def get_zattrs():
    return {
        "multiscales": [
            {
                "axes": [
                    {"name": "T", "type": "time", "unit": "second"},
                    {"name": "C", "type": "channel"},
                    {"name": "Z", "type": "space", "unit": "micrometer"},
                    {"name": "Y", "type": "space", "unit": "micrometer"},
                    {"name": "X", "type": "space", "unit": "micrometer"},
                ],
                "coordinateTransformations": [{"type": "identity"}],
                "datasets": [
                    {
                        "coordinateTransformations": [{"scale": s, "type": "scale"}],
                        "path": str(i),
                    }
                    for i, s in enumerate(SCALES)
                ],
                "name": "0",
                "version": "0.4",
            }
        ],
    }


@app.get("/.zgroup")
def get_zgroup():
    return {"zarr_format": 2}


@app.get("/{scale}/.zarray")
def get_zarray(scale: int):
    chunks, shape, _ = get_chunks_and_shape(scale)
    return {
        "chunks": chunks,
        "compressor": {
            "blocksize": 0,
            "clevel": 1,
            "cname": "zstd",
            "id": "blosc",
            "shuffle": 2,
        },
        "dimension_separator": "/",
        "dtype": "|u1",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": shape,
        "zarr_format": 2,
    }


@app.get("/{scale}/{t}/{c}/{z}/{y}/{x}")
def get_chunk(scale: int, t: int, c: int, z: int, y: int, x: int):
    chunk = generate_chunk(scale, t, c, z, y, x)
    compressed_chunk = BLOSC_ENCODER.encode(chunk)
    return fastapi.Response(
        content=compressed_chunk,
        media_type="application/octet-stream",
    )
