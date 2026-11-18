"""Opt-in receiver diagnostics; never participates in pose selection."""
import json
import os
import signal
import math
import queue
import threading
import time
from pathlib import Path


def alignment_details(poses, linear, angular, timestamp, mode, available, error=None):
    """Describe the existing lookup, using the same sorted-buffer convention."""
    times = [float(s[0]) for s in poses]
    import bisect
    right = bisect.bisect_left(times, timestamp)
    exact = right < len(times) and abs(times[right] - timestamp) <= 1e-9
    left_stamp = times[right - 1] if right else None
    right_stamp = times[right] if right < len(times) else None
    source = 'unavailable'
    reason = error
    selected = None
    if available:
        if exact:
            source, selected = 'exact', right_stamp
        elif right and right < len(times):
            source = 'interpolated'
        elif mode == 'endpoint':
            source = 'early_endpoint' if right == 0 else 'late_endpoint'
            selected = times[0] if right == 0 else times[-1]
        else:
            source, selected = 'extrapolated', times[-1]
    elif not reason:
        if not times:
            reason = 'empty_buffer'
        elif right == 0 and not exact:
            reason = 'before_buffer'
        elif not linear or not angular:
            reason = 'missing_twist' if mode == 'extrapolate' else 'outside_tolerance'
        else:
            reason = 'outside_tolerance_or_invalid_twist'
    endpoint = selected
    if endpoint is None and times and (right == 0 or right == len(times)):
        endpoint = times[0] if right == 0 else times[-1]
    return dict(alignment=mode, source=source, available=available, reason=reason,
                image_stamp=timestamp, odom_count=len(times),
                left_stamp=left_stamp, right_stamp=right_stamp,
                exact_stamp=right_stamp if exact else None,
                latest_stamp=times[-1] if times else None,
                signed_gap=timestamp-endpoint if endpoint is not None else None,
                left_gap=timestamp-left_stamp if left_stamp is not None else None,
                right_gap=timestamp-right_stamp if right_stamp is not None else None,
                linear_twist_stamp=float(linear[-1][0]) if linear else None,
                angular_twist_stamp=float(angular[-1][0]) if angular else None,
                linear_twist_age=timestamp-float(linear[-1][0]) if linear else None,
                angular_twist_age=timestamp-float(angular[-1][0]) if angular else None)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def process_identity(pid):
    """Linux start ticks distinguish the owning process from a reused PID."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def shutdown_summary(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-65536))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get('event') == 'writer_summary':
            return value
    return None


def finalize(path, timeout=15):
    """Signal the verified writer owner directly; never fabricate loss counters."""
    path = Path(path).resolve()
    metadata = Path(str(path)+'.shutdown.json')
    if metadata.exists():
        return json.loads(metadata.read_text())
    result = dict(log=str(path), signal_sent=False, summary_present=False, process_exited=False)
    try:
        summary = shutdown_summary(path)
        owner_path = Path(str(path)+'.owner.json')
        if not owner_path.exists():
            raise RuntimeError('Missing writer owner metadata; cannot safely signal a process')
        owner = json.loads(owner_path.read_text())
        pid = owner['pid']
        if (owner['log'] != str(path) or owner['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()):
            raise RuntimeError('Writer owner identity does not match this log/boot')
        if pid == os.getpid():
            raise RuntimeError('Refusing to signal this finalizer process')
        identity = process_identity(pid)
        if identity is not None and identity != owner['start_ticks']:
            raise RuntimeError('Stale writer owner: PID belongs to another process')
        alive = identity == owner['start_ticks'] and owner['start_ticks'] is not None
        if alive and summary is None:
            os.kill(pid, signal.SIGTERM)
            result['signal_sent'] = True
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            summary = shutdown_summary(path)
            alive = process_identity(pid) == owner['start_ticks'] and owner['start_ticks'] is not None
            if not alive:
                break
            time.sleep(.1)
        result.update(summary_present=summary is not None, process_exited=not alive, writer_summary=summary)
        if alive:
            raise RuntimeError('Writer owner did not exit before shutdown deadline')
        if summary is None or not summary.get('clean_shutdown', False):
            raise RuntimeError('Writer exited without a clean shutdown summary')
    except Exception as exc:
        result['error'] = str(exc)
    with metadata.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    return result


class ReceiverTiming:
    """Bounded nonblocking producer, single JSONL writer, explicit loss counters."""
    def __init__(self, path, capacity=4096):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', buffering=1)  # Never overwrite a prior capture.
        self.path = path
        owner = dict(pid=os.getpid(), start_ticks=process_identity(os.getpid()),
                     boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                     log=str(path.resolve()))
        try:
            with Path(str(path)+'.owner.json').open('x') as stream:
                json.dump(owner, stream)
        except Exception:
            self.file.close()
            raise
        self.written = 0
        self.queue = queue.Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.dropped = 0
        self.error = None
        self.sequence = 0
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def emit(self, event, **fields):
        if self.closed:
            return
        self.sequence += 1
        record = dict(event=event, sequence=self.sequence,
                      monotonic_ns=time.monotonic_ns(), **fields)
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                self.file.write(json.dumps(_finite(record), allow_nan=False) + '\n')
                self.written += 1
            self.file.write(json.dumps(dict(event='writer_summary',
                            dropped_records=self.dropped,
                            attempted_records=self.sequence, written_records=self.written,
                            clean_shutdown=True, writer_error=None, queue_remaining=self.queue.qsize(),
                            pid=os.getpid(), monotonic_ns=time.monotonic_ns())) + '\n')
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.file.close()

    def close(self):
        self.closed = True
        self.stopping.set()
        self.thread.join(timeout=5)
        return self.error or ('writer_shutdown_timeout' if self.thread.is_alive() else None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Finalize a verified receiver timing writer before killing its tmux session')
    parser.add_argument('--finalize', required=True)
    parser.add_argument('--timeout', type=float, default=15)
    args = parser.parse_args()
    result = finalize(args.finalize, args.timeout)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result.get('error') else 0)
