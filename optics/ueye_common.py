"""Shared IDS uEye camera setup/control helpers used by cmos_capture.py and
flat_preview.py.
"""
import ctypes
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from pyueye import ueye


def check(ret, msg="uEye call failed"):
    if ret != ueye.IS_SUCCESS:
        raise RuntimeError(f"{msg}. uEye ret={ret}")


def disable_auto(h_cam):
    zero = ueye.DOUBLE(0)
    check(ueye.is_SetAutoParameter(h_cam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, zero, None),
          "disable auto shutter")
    check(ueye.is_SetAutoParameter(h_cam, ueye.IS_SET_ENABLE_AUTO_GAIN, zero, None),
          "disable auto gain")


def set_exposure_ms(h_cam, exposure_ms: float):
    # Max exposure is capped by the frame time, so slow the frame rate down
    # first or the driver silently clamps long exposures to a few ms.
    t_min, t_max, t_inc = ueye.DOUBLE(), ueye.DOUBLE(), ueye.DOUBLE()
    check(ueye.is_GetFrameTimeRange(h_cam, t_min, t_max, t_inc), "is_GetFrameTimeRange")
    frame_time_s = min(max(exposure_ms / 1000.0, t_min.value), t_max.value)
    new_fps = ueye.DOUBLE()
    check(ueye.is_SetFrameRate(h_cam, ueye.DOUBLE(1.0 / frame_time_s), new_fps), "is_SetFrameRate")

    exp = ueye.DOUBLE(float(exposure_ms))
    ret = ueye.is_Exposure(h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp, ctypes.sizeof(exp))
    if ret != ueye.IS_SUCCESS:
        raise RuntimeError(f"Set exposure to {exposure_ms}ms failed. ret={ret}")


def get_exposure_ms(h_cam) -> float:
    exp = ueye.DOUBLE(0.0)
    check(ueye.is_Exposure(h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exp, ctypes.sizeof(exp)),
          "get exposure")
    return float(exp.value)


def get_camera_info(h_cam) -> dict:
    n = ueye.INT(0)
    check(ueye.is_GetNumberOfCameras(n), "is_GetNumberOfCameras")
    cam_list = ueye.UEYE_CAMERA_LIST(ueye.UEYE_CAMERA_INFO * max(n.value, 1))
    cam_list.dwCount = n.value
    check(ueye.is_GetCameraList(cam_list), "is_GetCameraList")
    ci = cam_list.uci[0]
    return {
        "model": bytes(ci.Model).split(b"\x00")[0].decode(),
        "serial": bytes(ci.SerNo).split(b"\x00")[0].decode(),
    }


def grab_frame(h_cam, mem_ptr, width, height, bits_per_pixel, pitch) -> np.ndarray:
    check(ueye.is_FreezeVideo(h_cam, ueye.IS_WAIT), "Freeze (grab)")
    img = ueye.get_data(mem_ptr, width, height, bits_per_pixel, pitch, copy=True)
    return np.reshape(img, (height.value, width.value))


@dataclass
class CameraHandle:
    h_cam: object
    mem_ptr: object
    mem_id: object
    width: object
    height: object
    bits_per_pixel: object
    pitch: object
    info: dict

    def set_exposure_ms(self, exposure_ms: float):
        set_exposure_ms(self.h_cam, exposure_ms)

    def get_exposure_ms(self) -> float:
        return get_exposure_ms(self.h_cam)

    def grab(self) -> np.ndarray:
        return grab_frame(self.h_cam, self.mem_ptr, self.width, self.height,
                           self.bits_per_pixel, self.pitch)


@contextmanager
def open_camera(color_mode=ueye.IS_CM_MONO8):
    """Init, configure, and allocate a uEye camera; tears everything back down on exit."""
    h_cam = ueye.HIDS(0)
    check(ueye.is_InitCamera(h_cam, None), "is_InitCamera")
    mem_ptr = None
    mem_id = None
    try:
        check(ueye.is_SetColorMode(h_cam, color_mode), "is_SetColorMode")
        disable_auto(h_cam)
        camera_info = get_camera_info(h_cam)

        rect_aoi = ueye.IS_RECT()
        check(ueye.is_AOI(h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ctypes.sizeof(rect_aoi)),
              "is_AOI GET")

        width = ueye.INT(int(rect_aoi.s32Width))
        height = ueye.INT(int(rect_aoi.s32Height))
        bits_per_pixel = ueye.INT(8)
        pitch = ueye.INT()

        mem_ptr = ueye.c_mem_p()
        mem_id = ueye.INT()
        check(ueye.is_AllocImageMem(h_cam, width, height, bits_per_pixel, mem_ptr, mem_id),
              "is_AllocImageMem")
        check(ueye.is_SetImageMem(h_cam, mem_ptr, mem_id), "is_SetImageMem")
        check(ueye.is_InquireImageMem(h_cam, mem_ptr, mem_id, width, height, bits_per_pixel, pitch),
              "is_InquireImageMem")

        # No is_CaptureVideo here: grabs use is_FreezeVideo (single-frame mode).
        # Running live video too makes FreezeVideo return IS_CAPTURE_RUNNING
        # immediately without waiting, so we'd read an unfilled (all-zero) buffer.
        yield CameraHandle(h_cam, mem_ptr, mem_id, width, height, bits_per_pixel, pitch, camera_info)
    finally:
        if mem_ptr is not None:
            ueye.is_StopLiveVideo(h_cam, ueye.IS_FORCE_VIDEO_STOP)
            ueye.is_FreeImageMem(h_cam, mem_ptr, mem_id)
        ueye.is_ExitCamera(h_cam)


def save_fits(path: str, frame: np.ndarray, exposure_ms: float, imagetyp: str,
              camera_info: dict, extra: dict = None) -> None:
    from astropy.io import fits
    from datetime import datetime, timezone

    hdu = fits.PrimaryHDU(frame.astype(np.uint8))
    hdu.header["EXPTIME"] = (exposure_ms / 1000.0, "Exposure time in seconds")
    hdu.header["DATE-OBS"] = datetime.now(timezone.utc).isoformat()
    hdu.header["IMAGETYP"] = imagetyp
    hdu.header["CAMERA"] = camera_info["model"]
    hdu.header["SERIALNO"] = camera_info["serial"]
    if extra:
        for key, value in extra.items():
            hdu.header[key] = value
    hdu.writeto(path, overwrite=True)
