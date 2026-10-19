"""
MicroLYNX stage server: run this in WSL (Linux), where the USB-serial adapter works.

    python3 stage_server.py /dev/ttyUSB0            # listens on port 5555

The Windows-side stage_focus_test.py connects with  --port tcp:localhost:5555
and sends one JSON request per line:

    {"op": "query", "cmd": "PRINT POS"}  -> {"ok": true, "resp": "12.5"}
    {"op": "mova",  "units": 12.5}       -> {"ok": true, "pos": "12.5"}   (returns after motion stops)
    {"op": "sstp"}                       -> {"ok": true}

Only one client at a time. If the client disconnects mid-run the stage is
soft-stopped. Needs only pyserial (pip install pyserial).
"""
import argparse
import json
import socket
import sys
import time

MOVE_TIMEOUT_S = 60.0


class Lynx:
    def __init__(self, ser):
        self.ser = ser
        time.sleep(0.2)
        self.ser.reset_input_buffer()

    def xfer(self, cmd, wait=0.05):
        self.ser.write((cmd + "\r").encode())
        time.sleep(wait)
        deadline = time.time() + 1.0
        buf = b""
        while time.time() < deadline:
            buf += self.ser.read_all() or b""
            if buf.rstrip().endswith(b">") or b"?" in buf:
                break
            time.sleep(0.01)
        return buf.decode(errors="ignore")

    def query(self, cmd):
        lines = self.xfer(cmd).replace("\r", "").split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and line != ">" and line.upper() != cmd.strip().upper():
                return line.lstrip(">").strip()
        return ""

    def mova(self, units, tol=0.0005):
        """Move and return once the stage has stopped.

        This controller reports MVG as TRUE/FALSE and has been seen reading TRUE
        while idle, so MVG alone isn't trusted: the move is done when POS is at
        the target and unchanged for 3 polls, or (fallback) POS has not changed
        for 10 polls.
        """
        target = float(units)
        self.xfer(f"MOVA {target:.6f}")
        t0 = time.time()
        deadline = t0 + MOVE_TIMEOUT_S
        last, stable = None, 0
        while time.time() < deadline:
            time.sleep(0.05)
            try:
                pos = float(self.query("PRINT POS"))
            except ValueError:
                continue
            stable = stable + 1 if last is not None and abs(pos - last) < 1e-6 else 0
            last = pos
            mvg = self.query("PRINT MVG").upper()
            at_target = abs(pos - target) < tol
            if at_target and (stable >= 3 or mvg in ("0", "FALSE")):
                return f"{pos}"
            if stable >= 10 and time.time() - t0 > 0.5:
                return f"{pos}"      # stopped short of target; caller compares
        self.xfer("SSTP")
        raise RuntimeError("motion timeout; soft-stopped")


def handle(lynx, req):
    op = req.get("op")
    if op == "query":
        cmd = str(req["cmd"])
        if not cmd.upper().startswith("PRINT "):
            raise ValueError("query only accepts PRINT commands")
        return {"resp": lynx.query(cmd)}
    if op == "mova":
        return {"pos": lynx.mova(req["units"])}
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
    a = p.parse_args()
    import serial
    ser = serial.Serial(port=a.device, baudrate=9600, bytesize=8, parity="N", stopbits=1, timeout=0.5)
    lynx = Lynx(ser)
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
