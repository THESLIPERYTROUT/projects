"""Standalone IDS uEye capture library.

    from ueye_cam import UEyeCamera, save_fits

    with UEyeCamera(exposure_ms=0.5) as cam:
        for frame in cam.burst(20):
            frame.save_fits(f"dark_{frame.index:02d}.fits", imagetyp="DARK")

Also runnable as a CLI: ``python -m ueye_cam --help``.
"""
from .camera import (
    COLOR_MODES,
    CameraError,
    CameraInfo,
    CameraTimeout,
    ExposureClampedWarning,
    Frame,
    FrameStream,
    UEyeCamera,
)
from .fits_io import frame_header, save_cube, save_fits

__all__ = [
    "COLOR_MODES", "CameraError", "CameraInfo", "CameraTimeout", "ExposureClampedWarning",
    "Frame", "FrameStream", "UEyeCamera", "frame_header", "save_cube", "save_fits",
]
