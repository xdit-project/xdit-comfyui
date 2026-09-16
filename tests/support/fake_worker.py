"""CPU-only child process that speaks the xDiT worker socket protocol."""

from __future__ import annotations

import subprocess
import sys
import textwrap

FAKE_WORKER = textwrap.dedent("""
    import json, pickle, socket, struct, sys, os
    from pathlib import Path

    socket_path, config_path = sys.argv[1], sys.argv[2]
    behaviour = os.environ.get("FAKE_WORKER_BEHAVIOUR", "serve")
    if behaviour == "die":
        print("worker died: ValueError: no such model", flush=True)
        raise SystemExit(1)
    if behaviour == "hang":
        while True:
            pass

    Path(socket_path).unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    Path(f"{socket_path}.ready").write_text("ok")
    print("worker ready", flush=True)


    def read_message(conn):
        (size,) = struct.unpack("!I", conn.recv(4))
        return json.loads(conn.recv(size).decode("utf-8"))


    while True:
        conn, _ = server.accept()
        job = read_message(conn)
        op = job.get("op")
        if op == "shutdown":
            conn.close()
            break
        if op == "park":
            reply = json.dumps({"ok": True, "host_bytes": 1000, "gpu_bytes": 100}).encode()
            conn.sendall(struct.pack("!I", len(reply)) + reply)
            conn.close()
            continue
        if op == "restore":
            reply = json.dumps({"ok": True, "gpu_bytes": 2048}).encode()
            conn.sendall(struct.pack("!I", len(reply)) + reply)
            conn.close()
            continue
        if behaviour == "hang_run":
            while True:
                pass
        payload = pickle.dumps(
            {
                "output": {"echo": job.get("config", {}).get("prompt")},
                "timings": [0.25],
                "metadata": {"fps": 24, "preprocessed_height": 512},
            }
        )
        conn.sendall(struct.pack("!I", len(payload)) + payload)
        conn.close()
    server.close()
    Path(socket_path).unlink(missing_ok=True)
    Path(f"{socket_path}.ready").unlink(missing_ok=True)
    """)

REAL_POPEN = subprocess.Popen


def spawn_fake_worker(cmd, **kwargs):
    """Replace a torchrun launch with a child that speaks the worker protocol."""
    socket_path, config_path = cmd[-2], cmd[-1]
    return REAL_POPEN(
        [
            sys.executable,
            "-c",
            FAKE_WORKER,
            socket_path,
            config_path,
            "torch.distributed.run -m xdit_comfyui.worker_server",
        ],
        **kwargs,
    )


def worker_init_config(model="black-forest-labs/FLUX.1-dev"):
    return {
        "model": model,
        "ulysses_degree": 1,
        "attention_backend": "sdpa",
        "prompt": "warm",
        "num_inference_steps": 4,
        "height": 512,
        "width": 512,
    }
