"""IDS uEye camera wrapper.

All pyueye/ctypes handling lives here. Callers get numpy frames wrapped in
``Frame`` objects carrying the metadata needed to write calibrated FITS files.

Capture modes:
    cam.snap()          one frame (single-frame trigger, is_FreezeVideo)
    cam.burst(n)        n frames back-to-back in single-frame mode
    cam.stream(...)     continuous live video through a ring buffer
"""
import ctypes
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, List, Optional

import numpy as np
from pyueye import ueye


class CameraError(RuntimeError):
    def __init__(self, msg: str, ret: Optional[int] = None):
        super().__init__(msg if ret is None else f"{msg} (uEye ret={ret})")
        self.ret = ret


class CameraTimeout(CameraError):
    pass


class ExposureClampedWarning(UserWarning):
    pass


def _check(ret, msg):
    if ret != ueye.IS_SUCCESS:
        raise CameraError(msg, ret)


def _cstr(raw) -> str:
    return bytes(raw).split(b"\x00")[0].decode(errors="replace")


# name -> (uEye color mode, bits per pixel in the buffer, significant data bits)
COLOR_MODES = {
    "mono8": (ueye.IS_CM_MONO8, 8, 8),
    "mono10": (ueye.IS_CM_MONO10, 16, 10),
    "mono12": (ueye.IS_CM_MONO12, 16, 12),
    "mono16": (ueye.IS_CM_MONO16, 16, 16),
}


@dataclass(frozen=True)
class CameraInfo:
    model: str
    serial: str
    width: int
    height: int
    pixel_size_um: float


@dataclass(frozen=True)
class Frame:
    data: np.ndarray        # (height, width), uint8 or uint16
    exposure_ms: float      # exposure the driver actually applied
    gain: int               # hardware master gain, 0-100
    bit_depth: int          # significant bits per pixel
    timestamp: datetime     # UTC, host clock at readout
    index: int              # position within the snap/burst/stream call
    camera: CameraInfo

    @property
    def saturation_value(self) -> int:
        return (1 << self.bit_depth) - 1

    @property
    def n_saturated(self) -> int:
        return int((self.data >= self.saturation_value).sum())

    def save_fits(self, path: str, imagetyp: str = "LIGHT", extra: dict = None) -> None:
        from .fits_io import save_fits
        save_fits(path, self, imagetyp=imagetyp, extra=extra)


class UEyeCamera:
    """Context-managed uEye camera.

        with UEyeCamera(exposure_ms=1.0) as cam:
            frame = cam.snap()
            frame.save_fits("light.fits")

    Auto shutter and auto gain are always disabled. After any exposure/gain
    change the next snap/burst discards one frame, since the frame already in
    flight was exposed with the old settings.
    """

    def __init__(self, camera_id: int = 0, color_mode: str = "mono8",
                 exposure_ms: Optional[float] = None, gain: Optional[int] = None):
        if color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {sorted(COLOR_MODES)}")
        self._camera_id = camera_id
        self._color_mode, self._bpp, self.bit_depth = COLOR_MODES[color_mode]
        self._dtype = np.dtype(np.uint8) if self._bpp == 8 else np.dtype("<u2")
        self._initial_exposure_ms = exposure_ms
        self._initial_gain = gain

        self._h_cam = None
        self._mem_ptr = None
        self._mem_id = None
        self._width = self._height = self._pitch = 0
        self._exposure_ms = 0.0
        self._gain = 0
        self._settings_changed = True
        self._active_stream = None
        self.info: Optional[CameraInfo] = None

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> "UEyeCamera":
        if self._h_cam is not None:
            return self
        h_cam = ueye.HIDS(self._camera_id)
        _check(ueye.is_InitCamera(h_cam, None), f"is_InitCamera (camera id {self._camera_id})")
        self._h_cam = h_cam
        try:
            _check(ueye.is_SetColorMode(h_cam, self._color_mode), "is_SetColorMode")
            zero = ueye.DOUBLE(0)
            _check(ueye.is_SetAutoParameter(h_cam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, zero, None),
                   "disable auto shutter")
            _check(ueye.is_SetAutoParameter(h_cam, ueye.IS_SET_ENABLE_AUTO_GAIN, zero, None),
                   "disable auto gain")

            rect = ueye.IS_RECT()
            _check(ueye.is_AOI(h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect, ctypes.sizeof(rect)),
                   "is_AOI get")
            self._width, self._height = int(rect.s32Width), int(rect.s32Height)

            # No is_CaptureVideo here: snaps use is_FreezeVideo (single-frame mode).
            # Running live video too makes FreezeVideo return IS_CAPTURE_RUNNING
            # immediately without waiting, so we'd read an unfilled (all-zero) buffer.
            self._mem_ptr, self._mem_id = self._alloc_buffer()
            _check(ueye.is_SetImageMem(h_cam, self._mem_ptr, self._mem_id), "is_SetImageMem")

            self.info = self._read_info()
            if self._initial_exposure_ms is not None:
                self.set_exposure_ms(self._initial_exposure_ms)
            else:
                self._exposure_ms = self._read_exposure_ms()
            if self._initial_gain is not None:
                self.set_gain(self._initial_gain)
            else:
                self._gain = self._read_gain()
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._h_cam is None:
            return
        if self._active_stream is not None:
            self._active_stream.close()
        ueye.is_StopLiveVideo(self._h_cam, ueye.IS_FORCE_VIDEO_STOP)
        if self._mem_ptr is not None:
            ueye.is_FreeImageMem(self._h_cam, self._mem_ptr, self._mem_id)
            self._mem_ptr = self._mem_id = None
        ueye.is_ExitCamera(self._h_cam)
        self._h_cam = None

    def __enter__(self) -> "UEyeCamera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def h_cam(self):
        """Raw pyueye handle, for driver calls this wrapper doesn't cover."""
        self._require_open()
        return self._h_cam

    # -- settings ------------------------------------------------------------

    @property
    def exposure_ms(self) -> float:
        return self._exposure_ms

    @exposure_ms.setter
    def exposure_ms(self, value: float) -> None:
        self.set_exposure_ms(value)

    def set_exposure_ms(self, exposure_ms: float) -> float:
        """Set exposure and return the value the driver actually applied.

        Warns with ExposureClampedWarning if that differs from the request by >1%.
        """
        self._require_open()
        # Max exposure is capped by the frame time, so slow the frame rate down
        # first or the driver silently clamps long exposures to a few ms.
        t_min, t_max, t_inc = ueye.DOUBLE(), ueye.DOUBLE(), ueye.DOUBLE()
        _check(ueye.is_GetFrameTimeRange(self._h_cam, t_min, t_max, t_inc), "is_GetFrameTimeRange")
        frame_time_s = min(max(exposure_ms / 1000.0, t_min.value), t_max.value)
        new_fps = ueye.DOUBLE()
        _check(ueye.is_SetFrameRate(self._h_cam, ueye.DOUBLE(1.0 / frame_time_s), new_fps),
               "is_SetFrameRate")

        exp = ueye.DOUBLE(float(exposure_ms))
        _check(ueye.is_Exposure(self._h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp, ctypes.sizeof(exp)),
               f"set exposure to {exposure_ms} ms")

        self._exposure_ms = self._read_exposure_ms()
        self._settings_changed = True
        if abs(self._exposure_ms - exposure_ms) > 0.01 * exposure_ms:
            warnings.warn(f"requested {exposure_ms} ms exposure, camera applied "
                          f"{self._exposure_ms:.4f} ms", ExposureClampedWarning, stacklevel=2)
        return self._exposure_ms

    @property
    def gain(self) -> int:
        return self._gain

    @gain.setter
    def gain(self, value: int) -> None:
        self.set_gain(value)

    def set_gain(self, gain: int) -> int:
        """Set hardware master gain (0-100) and return the applied value."""
        self._require_open()
        ign = ueye.IS_IGNORE_PARAMETER
        _check(ueye.is_SetHardwareGain(self._h_cam, int(gain), ign, ign, ign), f"set gain to {gain}")
        self._gain = self._read_gain()
        self._settings_changed = True
        return self._gain

    # -- capture -------------------------------------------------------------

    def snap(self) -> Frame:
        """Capture a single frame."""
        return self.burst(1)[0]

    def burst(self, n_frames: int) -> List[Frame]:
        """Capture n frames back-to-back in single-frame mode (e.g. a dark/flat series)."""
        self._require_open()
        if self._active_stream is not None:
            raise CameraError("can't snap while a stream is running")
        if self._settings_changed:
            self._freeze()  # discard the frame exposed with the old settings
            self._settings_changed = False
        frames = []
        for i in range(n_frames):
            data = self._freeze()
            frames.append(self._make_frame(data, i))
        return frames

    def stream(self, n_frames: Optional[int] = None, n_buffers: int = 4,
               timeout_ms: Optional[int] = None) -> "FrameStream":
        """Continuous live capture. Use as a context manager so the camera is
        always returned to single-frame mode:

            with cam.stream() as frames:
                for frame in frames:
                    ...

        Stops after n_frames if given, otherwise runs until closed. Frames come
        at roughly 1/exposure; if the consumer is slower, old buffers are reused
        and frames are dropped rather than queued.
        """
        self._require_open()
        if self._active_stream is not None:
            raise CameraError("a stream is already running")
        if timeout_ms is None:
            timeout_ms = int(3 * self._exposure_ms) + 1000
        return FrameStream(self, n_frames, n_buffers, timeout_ms)

    # -- internals -----------------------------------------------------------

    def _require_open(self):
        if self._h_cam is None:
            raise CameraError("camera is not open")

    def _alloc_buffer(self):
        mem_ptr, mem_id = ueye.c_mem_p(), ueye.INT()
        _check(ueye.is_AllocImageMem(self._h_cam, self._width, self._height, self._bpp, mem_ptr, mem_id),
               "is_AllocImageMem")
        width, height, bpp, pitch = ueye.INT(), ueye.INT(), ueye.INT(), ueye.INT()
        _check(ueye.is_InquireImageMem(self._h_cam, mem_ptr, mem_id, width, height, bpp, pitch),
               "is_InquireImageMem")
        self._pitch = int(pitch.value)
        return mem_ptr, mem_id

    def _copy_buffer(self, mem_ptr) -> np.ndarray:
        # Rows are padded out to `pitch` bytes; strip the padding before viewing
        # as pixels.
        raw = ueye.get_data(mem_ptr, self._width, self._height, self._bpp, self._pitch, copy=True)
        row_bytes = self._width * self._dtype.itemsize
        rows = raw.reshape(self._height, self._pitch)[:, :row_bytes]
        return np.ascontiguousarray(rows).view(self._dtype).reshape(self._height, self._width)

    def _freeze(self) -> np.ndarray:
        _check(ueye.is_FreezeVideo(self._h_cam, ueye.IS_WAIT), "is_FreezeVideo")
        return self._copy_buffer(self._mem_ptr)

    def _make_frame(self, data: np.ndarray, index: int) -> Frame:
        return Frame(data=data, exposure_ms=self._exposure_ms, gain=self._gain,
                     bit_depth=self.bit_depth, timestamp=datetime.now(timezone.utc),
                     index=index, camera=self.info)

    def _read_exposure_ms(self) -> float:
        exp = ueye.DOUBLE(0.0)
        _check(ueye.is_Exposure(self._h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exp, ctypes.sizeof(exp)),
               "get exposure")
        return float(exp.value)

    def _read_gain(self) -> int:
        ign = ueye.IS_IGNORE_PARAMETER
        return int(ueye.is_SetHardwareGain(self._h_cam, ueye.IS_GET_MASTER_GAIN, ign, ign, ign))

    def _read_info(self) -> CameraInfo:
        cam_info = ueye.CAMINFO()
        _check(ueye.is_GetCameraInfo(self._h_cam, cam_info), "is_GetCameraInfo")
        sensor = ueye.SENSORINFO()
        _check(ueye.is_GetSensorInfo(self._h_cam, sensor), "is_GetSensorInfo")
        pixel_size = getattr(sensor.wPixelSize, "value", sensor.wPixelSize)
        return CameraInfo(model=_cstr(sensor.strSensorName), serial=_cstr(cam_info.SerNo),
                          width=self._width, height=self._height,
                          pixel_size_um=int(pixel_size) / 100.0)


class FrameStream:
    """Live-video iterator returned by UEyeCamera.stream()."""

    def __init__(self, cam: UEyeCamera, n_frames: Optional[int], n_buffers: int, timeout_ms: int):
        self._cam = cam
        self._n_frames = n_frames
        self._timeout_ms = timeout_ms
        self._count = 0
        self._buffers = []
        self._running = False

        h_cam = cam._h_cam
        try:
            for _ in range(n_buffers):
                mem_ptr, mem_id = cam._alloc_buffer()
                self._buffers.append((mem_ptr, mem_id))
                _check(ueye.is_AddToSequence(h_cam, mem_ptr, mem_id), "is_AddToSequence")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                _check(ueye.is_InitImageQueue(h_cam, 0), "is_InitImageQueue")
            cam._active_stream = self
            self._running = True
            _check(ueye.is_CaptureVideo(h_cam, ueye.IS_DONT_WAIT), "is_CaptureVideo")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "FrameStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        if not self._running or (self._n_frames is not None and self._count >= self._n_frames):
            self.close()
            raise StopIteration
        h_cam = self._cam._h_cam
        mem_ptr, mem_id = ueye.c_mem_p(), ueye.INT()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ret = ueye.is_WaitForNextImage(h_cam, self._timeout_ms, mem_ptr, mem_id)
        if ret == ueye.IS_TIMED_OUT:
            raise CameraTimeout(f"no frame within {self._timeout_ms} ms", ret)
        _check(ret, "is_WaitForNextImage")
        try:
            data = self._cam._copy_buffer(mem_ptr)
        finally:
            ueye.is_UnlockSeqBuf(h_cam, mem_id, mem_ptr)
        frame = self._cam._make_frame(data, self._count)
        self._count += 1
        return frame

    def close(self) -> None:
        cam = self._cam
        h_cam = cam._h_cam
        if h_cam is None:
            return
        if self._running:
            ueye.is_StopLiveVideo(h_cam, ueye.IS_FORCE_VIDEO_STOP)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                ueye.is_ExitImageQueue(h_cam)
            self._running = False
        if self._buffers:
            ueye.is_ClearSequence(h_cam)
            for mem_ptr, mem_id in self._buffers:
                ueye.is_FreeImageMem(h_cam, mem_ptr, mem_id)
            self._buffers = []
            # Hand the single-frame buffer back so snap()/burst() work again.
            _check(ueye.is_SetImageMem(h_cam, cam._mem_ptr, cam._mem_id), "is_SetImageMem")
        if cam._active_stream is self:
            cam._active_stream = None
