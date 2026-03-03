import ctypes
import numpy as np
from pyueye import ueye

def check(ret, msg="uEye call failed"):
    # Some calls (e.g. is_FreezeVideo) return IS_CAPTURE_RUNNING (140) when the
    # camera is actively capturing.  Treat that as success so the script doesn't
    # throw a RuntimeError on a normal state.
    if ret not in (ueye.IS_SUCCESS, ueye.IS_CAPTURE_RUNNING):
        raise RuntimeError(f"{msg}. uEye ret={ret}")

def disable_auto(h_cam):
    zero = ueye.DOUBLE(0)
    check(ueye.is_SetAutoParameter(h_cam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, zero, None),
          "disable auto shutter")
    check(ueye.is_SetAutoParameter(h_cam, ueye.IS_SET_ENABLE_AUTO_GAIN, zero, None),
          "disable auto gain")

def set_exposure_ms(h_cam, exposure_ms: float):
    exp = ueye.DOUBLE(float(exposure_ms))
    ret = ueye.is_Exposure(h_cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp, ctypes.sizeof(exp))
    if ret != ueye.IS_SUCCESS:
        raise RuntimeError(f"Set exposure failed. ret={ret}")

def get_exposure_ms(h_cam) -> float:
    exp = ueye.DOUBLE(0.0)
    check(ueye.is_Exposure(h_cam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exp, ctypes.sizeof(exp)),
          "get exposure")
    return float(exp.value)

def main():
    h_cam = ueye.HIDS(0)
    check(ueye.is_InitCamera(h_cam, None), "is_InitCamera")
    try:
        check(ueye.is_SetColorMode(h_cam, ueye.IS_CM_MONO8), "is_SetColorMode")
        disable_auto(h_cam)

        # Get AOI
        rect_aoi = ueye.IS_RECT()
        check(ueye.is_AOI(h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ctypes.sizeof(rect_aoi)),
              "is_AOI GET")

        width  = ueye.INT(int(rect_aoi.s32Width))
        height = ueye.INT(int(rect_aoi.s32Height))
        bits_per_pixel = ueye.INT(8)
        pitch = ueye.INT()

        mem_ptr = ueye.c_mem_p()
        mem_id  = ueye.INT()

        # Allocate once
        check(ueye.is_AllocImageMem(h_cam, width, height, bits_per_pixel, mem_ptr, mem_id),
              "is_AllocImageMem")
        check(ueye.is_SetImageMem(h_cam, mem_ptr, mem_id), "is_SetImageMem")
        check(ueye.is_InquireImageMem(h_cam, mem_ptr, mem_id, width, height, bits_per_pixel, pitch),
              "is_InquireImageMem")

        # Start capture once (helps settings apply consistently)
        check(ueye.is_CaptureVideo(h_cam, ueye.IS_DONT_WAIT), "is_CaptureVideo")

        exposures = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
        for exposure_ms in exposures:
            set_exposure_ms(h_cam, exposure_ms)
            applied = get_exposure_ms(h_cam)
            print(f"Requested {exposure_ms} ms, applied {applied:.6f} ms")

            # Flush + grab
            check(ueye.is_FreezeVideo(h_cam, ueye.IS_WAIT), "Freeze (flush)")
            check(ueye.is_FreezeVideo(h_cam, ueye.IS_WAIT), "Freeze (grab)")

            img = ueye.get_data(mem_ptr, width, height, bits_per_pixel, pitch, copy=True)
            frame = np.reshape(img, (height.value, width.value))

            mx = int(frame.max())
            sat = int((frame == 255).sum())
            print("  max pixel:", mx, " saturated px:", sat)

        # Cleanup
        ueye.is_StopLiveVideo(h_cam, ueye.IS_FORCE_VIDEO_STOP)
        ueye.is_FreeImageMem(h_cam, mem_ptr, mem_id)

    finally:
        ueye.is_ExitCamera(h_cam)

if __name__ == "__main__":
    main()