"""
MicroLYNX stage server: run this in WSL (Linux), where the USB-serial adapter works.

    python3 stage_server.py /dev/ttyUSB0            # listens on port 5555

The Windows-side stage_focus_test.py connects with  --port tcp:localhost:5555
and sends one JSON request per line:

    {"op": "query", "cmd": "PRINT POS"}  -> {"ok": true, "resp": "12.5"}
    {"op": "mova",  "units": 12.5}       -> {"ok": true, "pos": "12.5", "reached": true}
                                            (returns after motion stops; reached=false if it
                                             stopped elsewhere, e.g. on a limit)
    {"op": "sstp"}                       -> {"ok": true}

Only one client at a time. If the client disconnects mid-run the stage is
soft-stopped. Needs only pyserial (pip install pyserial). Add --verbose to
print every position poll during a move (for debugging moves that hang).
"""
import argparse
import json
import socket
import sys
import time

MOVE_TIMEOUT_S = 60.0


def _is_value(line, cmd):
    """True for a reply line that carries a value (not blank, prompt or echo)."""
    line = line.strip().strip(">").strip()
    return bool(line) and line.upper() != cmd.strip().upper()


class Lynx:
    def __init__(self, ser, verbose=False):
        self.ser = ser
        self.verbose = verbose
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def xfer(self, cmd, timeout=1.0):
        """Send one command and return the raw reply.

        Any stale bytes (e.g. a reply that arrived after an earlier call gave
        up) are dropped first, otherwise every later reply is read one command
        out of step -- POS answers with MVG's value and vice versa. A PRINT
        returns as soon as its value line is in, so a controller that doesn't
        send the '>' prompt doesn't cost the full timeout on every query.
        """
        self.ser.reset_input_buffer()
        self.ser.write((cmd + "\r").encode())
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            buf += self.ser.read_all() or b""
            if self._reply_complete(cmd, buf):
                break
            time.sleep(0.005)
        return buf.decode(errors="ignore")

    @staticmethod
    def _reply_complete(cmd, buf):
        if buf.rstrip().endswith(b">") or b"?" in buf:
            return True
        if not cmd.upper().startswith("PRINT"):
            return False
        done_lines = buf.replace(b"\r", b"").split(b"\n")[:-1]   # lines with their newline received
        return any(_is_value(line.decode(errors="ignore"), cmd) for line in done_lines)

    def query(self, cmd):
        lines = self.xfer(cmd).replace("\r", "").split("\n")
        for line in reversed(lines):
            if _is_value(line, cmd):
                return line.strip().strip(">").strip()
        return ""

    def position(self):
        """POS as a float, or None if the reply couldn't be parsed."""
        try:
            return float(self.query("PRINT POS"))
        except ValueError:
            return None

    def mova(self, units, tol=0.0005, still_tol=None, settle_s=0.25, stall_s=1.5):
        """Move and return {"pos", "reached"} once the stage has stopped.

        MVG isn't used: on this controller it has been seen reading TRUE while
        idle. Instead the stage counts as stopped once POS has stayed within
        `still_tol` (default: tol, so encoder dither of a count or two doesn't
        look like motion) for `settle_s` at the target, or for `stall_s`
        anywhere (limit switch, stall, rejected move) -- then reached=False.
        Timing is in seconds, not polls, so slow serial replies don't stretch it.
        """
        target = float(units)
        still_tol = tol if still_tol is None else still_tol
        reply = self.xfer(f"MOVA {target:.6f}", timeout=0.3)
        if "?" in reply:
            raise RuntimeError(f"controller rejected MOVA {target}: {reply.strip()!r}")
        t0 = time.time()
        deadline = t0 + MOVE_TIMEOUT_S
        anchor, anchor_t, pos = None, t0, None
        while time.time() < deadline:
            pos = self.position()
            now = time.time()
            if pos is None:
                continue
            if anchor is None or abs(pos - anchor) > still_tol:
                anchor, anchor_t = pos, now          # moved: restart the stillness clock
            still_for = now - anchor_t
            if self.verbose:
                print(f"  t={now - t0:6.2f}s pos={pos:.6f} still={still_for:.2f}s")
            if abs(pos - target) <= tol and still_for >= settle_s:
                return {"pos": f"{pos}", "reached": True}
            if still_for >= stall_s:
                return {"pos": f"{pos}", "reached": False}
            time.sleep(0.02)
        self.xfer("SSTP")
        raise RuntimeError(f"motion timeout after {MOVE_TIMEOUT_S:.0f}s; soft-stopped "
                           f"(last pos {pos}, target {target})")


def handle(lynx, req):
    op = req.get("op")
    if op == "query":
        cmd = str(req["cmd"])
        if not cmd.upper().startswith("PRINT "):
            raise ValueError("query only accepts PRINT commands")
        return {"resp": lynx.query(cmd)}
    if op == "mova":
        return lynx.mova(req["units"], tol=float(req.get("tol", 0.0005)))
    if op == "sstp":
        lynx.xfer("SSTP")
        return {}
    raise ValueError(f"unknown op {op!r}")


def serve(lynx, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"stage server listening on {host}:{port}  (Ctrl+C to quit)")
    while True:
        conn, addr = srv.accept()
        print(f"client connected from {addr[0]}")
        f = conn.makefile("rwb")
        try:
            for raw in f:
                try:
                    req = json.loads(raw)
                    out = {"ok": True, **handle(lynx, req)}
                except Exception as exc:
                    out = {"ok": False, "error": str(exc)}
                f.write((json.dumps(out) + "\n").encode())
                f.flush()
        except (ConnectionError, OSError):
            pass
        finally:
            lynx.xfer("SSTP")
            conn.close()
            print("client disconnected (stage soft-stopped)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("device", nargs="?", default="/dev/ttyUSB0")
    p.add_argument("--host", default="0.0.0.0",
                   help="bind address; 0.0.0.0 so Windows can reach it through WSL localhost forwarding")
    p.add_argument("--tcp-port", type=int, default=5555)
    p.add_argument("--verbose", action="store_true", help="print each position poll while moving")
    a = p.parse_args()
    import serial
    ser = serial.Serial(port=a.device, baudrate=9600, bytesize=8, parity="N", stopbits=1, timeout=0.5)
    lynx = Lynx(ser, verbose=a.verbose)
    print(f"connected to {a.device}: POS={lynx.query('PRINT POS')}  MUNIT={lynx.query('PRINT MUNIT')}")
    try:
        serve(lynx, a.host, a.tcp_port)
    except KeyboardInterrupt:
        print()
    finally:
        lynx.xfer("SSTP")
        ser.close()


if __name__ == "__main__":
    main()
