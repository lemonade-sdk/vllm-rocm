#!/usr/bin/env python3
"""
Tier 2 — standalone functional inference (requires a real AMD GPU).

Starts the candidate bundle's own vllm-server (no lemonade) on a small model
and exercises the OpenAI-compatible endpoints. Uses a tiny text-only model so
the tier stays fast; real-model correctness and quantization are covered by
Tier 3 against the hot Lemonade models.

Checks:
  T2.1  server boot          — vllm-server starts and /health is ready (gating).
  T2.2  completion           — /v1/completions returns non-empty text (gating).
  T2.3  greedy determinism   — identical temp=0 requests give identical output
                               (gating; model-agnostic correctness signal).
  T2.4  chat                 — /v1/chat/completions returns content (gating).
  T2.5  streaming            — streamed completion yields chunks + [DONE] (gating).

Usage:
    python3 tier2_inference.py --bundle-root ./vllm-install --gfx-target gfx1151 \
        --model facebook/opt-125m --output tier2-gfx1151.json
"""

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import bundle
import report

READY_TIMEOUT = 600
REQUEST_TIMEOUT = 120


def http_post(url, payload, timeout=REQUEST_TIMEOUT, stream=False):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp  # caller iterates resp
    body = resp.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def http_get(url, timeout=10):
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status
    except (urllib.error.URLError, OSError):
        return None


def wait_ready(port, proc, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"server exited early (rc={proc.returncode})"
        if http_get(f"http://127.0.0.1:{port}/health") == 200:
            return True, None
        if http_get(f"http://127.0.0.1:{port}/v1/models") == 200:
            return True, None
        time.sleep(5)
    return False, f"not ready within {timeout}s"


def _tail(path, lines=40):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "\n".join(handle.read().splitlines()[-lines:])
    except OSError:
        return ""


class Server:
    def __init__(self, root, model, port, log_path):
        self.root = root
        self.model = model
        self.port = port
        self.log_path = log_path
        self.proc = None

    def start(self):
        launcher = os.path.join(self.root, "bin", "vllm-server")
        log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [
                launcher,
                "--model", self.model,
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "--dtype", "float16",
                "--max-model-len", "512",
                "--gpu-memory-utilization", "0.5",
                "--enforce-eager",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        # vLLM spawns child processes whose titles ("VLLM::EngineCore") don't
        # match "vllm.entrypoints"; case-insensitively sweep all of them so no
        # orphan keeps holding VRAM into the next tier/run.
        subprocess.run(
            ["pkill", "-9", "-if", "vllm|enginecore|resource_tracker"],
            check=False,
        )


# --------------------------------------------------------------------------
# Checks (closures bound to a running server in main)
# --------------------------------------------------------------------------


def check_completion(base, model):
    body = http_post(
        f"{base}/v1/completions",
        {"model": model, "prompt": "The capital of France is",
         "max_tokens": 16, "temperature": 0.0},
    )
    text = body.get("choices", [{}])[0].get("text", "")
    finish = body.get("choices", [{}])[0].get("finish_reason")
    if text.strip():
        return (report.STATUS_PASS, None, {"finish_reason": finish,
                                           "sample": text[:80]})
    return (report.STATUS_FAIL, "empty completion text", {"body": body})


def check_determinism(base, model):
    prompt = "Count: 1 2 3"
    out = []
    for _ in range(2):
        body = http_post(
            f"{base}/v1/completions",
            {"model": model, "prompt": prompt, "max_tokens": 24,
             "temperature": 0.0, "seed": 0},
        )
        out.append(body.get("choices", [{}])[0].get("text", ""))
    if out[0] and out[0] == out[1]:
        return (report.STATUS_PASS, None, {"sample": out[0][:80]})
    return (
        report.STATUS_FAIL,
        "greedy decode not deterministic across identical requests",
        {"first": out[0][:80], "second": out[1][:80]},
    )


def check_chat(base, model):
    body = http_post(
        f"{base}/v1/chat/completions",
        {"model": model,
         "messages": [{"role": "user", "content": "Say hello."}],
         "max_tokens": 16, "temperature": 0.0},
    )
    msg = body.get("choices", [{}])[0].get("message", {})
    content = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    if content.strip():
        return (report.STATUS_PASS, None, {"sample": content[:80]})
    return (report.STATUS_FAIL, "empty chat content", {"body": body})


def check_streaming(base, model):
    resp = http_post(
        f"{base}/v1/completions",
        {"model": model, "prompt": "Hello", "max_tokens": 16,
         "temperature": 0.0, "stream": True},
        stream=True,
    )
    chunks = 0
    saw_done = False
    for raw in resp:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            saw_done = True
            break
        chunks += 1
    if chunks > 0 and saw_done:
        return (report.STATUS_PASS, None, {"chunks": chunks})
    return (
        report.STATUS_FAIL,
        f"streaming incomplete (chunks={chunks}, done={saw_done})",
        {},
    )


def detect_versions(root):
    sp = bundle.find_site_packages(root)

    def ver(project):
        for meta in glob.glob(os.path.join(sp, f"{project}-*.dist-info", "METADATA")):
            with open(meta, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        return None

    torch_v = ver("torch")
    rocm_v = None
    if torch_v and "rocm" in torch_v:
        match = re.search(r"rocm([\d.]+)", torch_v)
        rocm_v = match.group(1) if match else None
    return ver("vllm"), torch_v, rocm_v


def main():
    parser = argparse.ArgumentParser(description="Tier 2 functional inference")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--gfx-target", required=True)
    parser.add_argument("--channel", default=None, choices=[None, "stable", "nightly"])
    # Small *instruct* model: supports /v1/chat/completions (has a chat
    # template) as well as /v1/completions, so every endpoint check is valid.
    # A base model like facebook/opt-125m 400s on the chat endpoint.
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--candidate-tag", default=None)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--logs-dir", default=".")
    parser.add_argument("--output", default="tier2-report.json")
    parser.add_argument("--ready-timeout", type=int, default=READY_TIMEOUT)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.bundle_root)
    vllm_v, torch_v, rocm_v = detect_versions(root)
    meta = report.build_meta(
        gfx_target=args.gfx_target,
        channel=args.channel,
        vllm_version=vllm_v,
        torch_version=torch_v,
        rocm_version=rocm_v,
        candidate_tag=args.candidate_tag,
        hardware_validated=True,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )
    tier = report.TierReport("tier2", meta)

    os.makedirs(args.logs_dir, exist_ok=True)
    log_path = os.path.join(args.logs_dir, f"vllm-server-{args.gfx_target}.log")
    base = f"http://127.0.0.1:{args.port}"
    server = Server(root, args.model, args.port, log_path)

    try:
        server.start()
        ready, err = wait_ready(args.port, server.proc, args.ready_timeout)
        if ready:
            tier.add("T2.1", "server boot", report.STATUS_PASS,
                     details={"model": args.model})
        else:
            tier.add("T2.1", "server boot", report.STATUS_FAIL, error=err,
                     details={"log_tail": _tail(log_path)})

        if ready:
            tier.run("T2.2", "completion", lambda: check_completion(base, args.model))
            tier.run("T2.3", "greedy determinism",
                     lambda: check_determinism(base, args.model))
            tier.run("T2.4", "chat", lambda: check_chat(base, args.model))
            tier.run("T2.5", "streaming", lambda: check_streaming(base, args.model))
        else:
            for tid, name in [("T2.2", "completion"), ("T2.3", "greedy determinism"),
                              ("T2.4", "chat"), ("T2.5", "streaming")]:
                tier.add(tid, name, report.STATUS_SKIP, error="server not ready")
    finally:
        server.stop()

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")

    if args.fail_on_error and tier.rollup_status() == report.STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
