"""
MicroLYNX linear stage CLI controller.

Usage:
    python NSC1_linear_stage_test.py [PORT]

Default port is /dev/ttyUSB0 (Linux) or COM3 (Windows). Override via argument.
"""

import sys
import time
import threading
import serial
from serial.tools import list_ports

DEFAULT_PORT = 'COM3' if sys.platform == 'win32' else '/dev/ttyUSB0'
PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT

POLL_INTERVAL  = 0.05   # seconds between MVG polls
MOVE_TIMEOUT   = 60.0   # seconds before giving up waiting for motion
LIMIT_POLL     = 0.15   # seconds between limit checks during slew

_ser_lock          = threading.Lock()
_slew_monitor_stop = threading.Event()


# --------------------------------------------------------------------------- #
# Serial helpers
# --------------------------------------------------------------------------- #

def send_cmd(cmd: str) -> str:
    with _ser_lock:
        ser.write((cmd + '\r').encode())
        time.sleep(POLL_INTERVAL)
        raw = ser.read_all()
    return raw.decode(errors='ignore').strip() if raw else ''


def query(cmd: str) -> str:
    """Send a command and return just the value line.

    The MicroLYNX echoes the command and appends a '>' prompt, e.g.:
        PRINT MVG\r\n0\r\n>
    This strips both so callers get the bare value ('0', '1', '200.0', …).
    """
    resp = send_cmd(cmd)
    lines = resp.replace('\r', '').split('\n')
    cmd_upper = cmd.strip().upper()
    for line in reversed(lines):
        line = line.strip()
        if line and line != '>' and line.upper() != cmd_upper:
            return line
    return resp.strip()


def wait_for_stop() -> None:
    """Wait until motion is complete.

    Primary: poll MVG flag for '0'.
    Fallback: if MVG never resolves cleanly, declare stopped once position
    reads the same value STABLE_NEEDED times in a row.
    """
    STABLE_NEEDED = 4
    deadline = time.time() + MOVE_TIMEOUT
    stable_count = 0
    last_pos: str | None = None

    while time.time() < deadline:
        # --- primary check ---
        mvg = query('PRINT MVG')
        if mvg == '0':
            return

        # --- fallback: position stability ---
        pos = query('PRINT POS')
        if pos == last_pos:
            stable_count += 1
            if stable_count >= STABLE_NEEDED:
                return
        else:
            stable_count = 0
            last_pos = pos

        time.sleep(POLL_INTERVAL)

    print('[WARN] Motion timeout — stage may still be moving.')


def get_pos() -> str:
    return query('PRINT POS')


def print_pos() -> None:
    print(f'  position: {get_pos()}')


def check_limits() -> None:
    """Query limit flags and print whatever state they report (for diagnostics)."""
    lmtp = query('PRINT LMTP')
    lmtm = query('PRINT LMTM')
    if lmtp == '1':
        print('  [!] POSITIVE limit switch active')
    elif lmtm == '1':
        print('  [!] NEGATIVE limit switch active')


def motion_done() -> None:
    print_pos()
    check_limits()


# --------------------------------------------------------------------------- #
# Motion commands
# --------------------------------------------------------------------------- #

def move_relative(steps: float) -> None:
    send_cmd(f'MOVR {steps}')
    wait_for_stop()
    motion_done()


def move_absolute(pos: float) -> None:
    send_cmd(f'MOVA {pos}')
    wait_for_stop()
    motion_done()


def _report_limit_hit() -> None:
    """Print limit-switch context after unexpected stop during slew.

    Primary: try LMTP / LMTM flags.
    Fallback: generic message so the user always gets some feedback even if
    the flag names or polarity differ from expectations.
    """
    lmtp = query('PRINT LMTP')
    lmtm = query('PRINT LMTM')
    if lmtp == '1':
        print('\n  [!] POSITIVE limit switch hit — type "stop" to continue')
    elif lmtm == '1':
        print('\n  [!] NEGATIVE limit switch hit — type "stop" to continue')
    else:
        # Flags inconclusive (wrong name, wrong polarity, not configured as
        # dedicated limits) — still tell the user something stopped the stage.
        print(f'\n  [!] Stage stopped unexpectedly (LMTP={lmtp} LMTM={lmtm})'
              ' — possible limit hit. Type "stop" to continue')


def _slew_limit_monitor() -> None:
    """Background thread: detect unexpected motion stop during a slew.

    Uses MVG flag as primary signal. Falls back to position stability.
    Only fires the report when the stop was NOT triggered by the user
    calling stop() (i.e. _slew_monitor_stop is still clear).
    """
    last_pos: str | None = None
    stable_count = 0
    STABLE_NEEDED = 3

    while not _slew_monitor_stop.wait(timeout=LIMIT_POLL):
        # Primary: MVG flag
        mvg = query('PRINT MVG')
        if mvg == '0':
            if not _slew_monitor_stop.is_set():
                _report_limit_hit()
            break

        # Fallback: position stability
        pos = query('PRINT POS')
        if pos == last_pos:
            stable_count += 1
            if stable_count >= STABLE_NEEDED:
                if not _slew_monitor_stop.is_set():
                    _report_limit_hit()
                break
        else:
            stable_count = 0
            last_pos = pos


def slew(speed: float) -> None:
    """Start constant-velocity slew; background thread watches for unexpected stop."""
    _slew_monitor_stop.clear()
    send_cmd(f'SLEW {speed}')
    print(f'  slewing at {speed} units/sec — type "stop" to halt')
    threading.Thread(target=_slew_limit_monitor, daemon=True).start()


def stop() -> None:
    _slew_monitor_stop.set()   # signal monitor before touching serial
    send_cmd('SSTP')
    wait_for_stop()
    motion_done()


def set_zero() -> None:
    send_cmd('POS=0')
    print('  position zeroed')
    print_pos()


# --------------------------------------------------------------------------- #
# Parameter helpers
# --------------------------------------------------------------------------- #

PARAM_MAP = {
    'vm':    'VM',
    'vi':    'VI',
    'accl':  'ACCL',
    'decl':  'DECL',
    'munit': 'MUNIT',
    'msel':  'MSEL',
}


def set_param(name: str, value: str) -> None:
    key = name.lower()
    if key not in PARAM_MAP:
        print(f'  unknown param "{name}". Options: {", ".join(PARAM_MAP)}')
        return
    reg = PARAM_MAP[key]
    send_cmd(f'{reg}={value}')
    print(f'  {reg} = {value}')


def print_status() -> None:
    for label, reg in [
        ('position ', 'POS'),
        ('vm       ', 'VM'),
        ('vi       ', 'VI'),
        ('accl     ', 'ACCL'),
        ('decl     ', 'DECL'),
        ('munit    ', 'MUNIT'),
    ]:
        val = query(f'PRINT {reg}')
        print(f'  {label}: {val}')


# --------------------------------------------------------------------------- #
# CLI loop
# --------------------------------------------------------------------------- #

HELP = """
Commands:
  mr  <n>           move relative N units
  ma  <n>           move absolute to position N
  sl  <speed>       slew at constant velocity (units/sec); negative = reverse
  stop              soft stop (decelerates)
  pos               print current position
  zero              set current position to zero
  set <param> <n>   set parameter (vm, vi, accl, decl, munit, msel)
  status            print all motion parameters
  help              show this message
  quit / exit       close connection and exit
"""


def run_cli() -> None:
    print(HELP)
    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ('quit', 'exit', 'q'):
                break

            elif cmd in ('help', '?'):
                print(HELP)

            elif cmd == 'mr' and len(parts) == 2:
                move_relative(float(parts[1]))

            elif cmd == 'ma' and len(parts) == 2:
                move_absolute(float(parts[1]))

            elif cmd in ('sl', 'slew') and len(parts) == 2:
                slew(float(parts[1]))

            elif cmd == 'stop':
                stop()

            elif cmd == 'pos':
                print_pos()

            elif cmd == 'zero':
                set_zero()

            elif cmd == 'set' and len(parts) == 3:
                set_param(parts[1], parts[2])

            elif cmd == 'status':
                print_status()

            else:
                print(f'  unrecognised command. Type "help" for options.')

        except ValueError as exc:
            print(f'  bad value: {exc}')
        except serial.SerialException as exc:
            print(f'  serial error: {exc}')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

serial_exception = getattr(serial, 'SerialException', Exception)

try:
    ser = serial.Serial(
        port=PORT,
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
    )
except serial_exception as exc:
    print(f'Could not open {PORT}: {exc}')
    ports = ', '.join(p.device for p in list_ports.comports()) or 'none found'
    print(f'Available ports: {ports}')
    sys.exit(1)

print(f'Connected to {PORT}')
try:
    run_cli()
finally:
    ser.close()
    print('Port closed.')
