"""Write uEye frames to FITS with consistent headers."""
from typing import Sequence

import numpy as np
from astropy.io import fits

from .camera import Frame


def frame_header(frame: Frame, imagetyp: str = "LIGHT", extra: dict = None) -> fits.Header:
    header = fits.Header()
    header["EXPTIME"] = (frame.exposure_ms / 1000.0, "Exposure time in seconds")
    header["DATE-OBS"] = (frame.timestamp.isoformat(), "UTC, host clock at readout")
    header["IMAGETYP"] = imagetyp
    header["GAIN"] = (frame.gain, "uEye hardware master gain (0-100)")
    header["BITDEPTH"] = (frame.bit_depth, "Significant bits per pixel")
    header["FRAMENUM"] = (frame.index, "Index within the capture sequence")
    header["CAMERA"] = frame.camera.model
    header["SERIALNO"] = frame.camera.serial
    header["XPIXSZ"] = (frame.camera.pixel_size_um, "Pixel width in microns")
    header["YPIXSZ"] = (frame.camera.pixel_size_um, "Pixel height in microns")
    if extra:
        for key, value in extra.items():
            header[key] = value
    return header


def save_fits(path: str, frame: Frame, imagetyp: str = "LIGHT", extra: dict = None,
              overwrite: bool = True) -> None:
    """Write one frame as a 2D image in its native dtype (uint8 or uint16)."""
    hdu = fits.PrimaryHDU(frame.data, header=frame_header(frame, imagetyp, extra))
    hdu.writeto(path, overwrite=overwrite)


def save_cube(path: str, frames: Sequence[Frame], imagetyp: str = "LIGHT", extra: dict = None,
              overwrite: bool = True) -> None:
    """Write a burst/stream as one 3D cube (frame, y, x). Header comes from the
    first frame plus NFRAMES; DATE-END is the last frame's timestamp."""
    if not frames:
        raise ValueError("no frames to save")
    header = frame_header(frames[0], imagetyp, extra)
    del header["FRAMENUM"]
    header["NFRAMES"] = (len(frames), "Frames in cube")
    header["DATE-END"] = frames[-1].timestamp.isoformat()
    hdu = fits.PrimaryHDU(np.stack([f.data for f in frames]), header=header)
    hdu.writeto(path, overwrite=overwrite)
