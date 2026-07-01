#!/usr/bin/env python3
"""
Tier 2 (omni variant) — standalone functional inference for vLLM-Omni bundles.

Runs INSTEAD OF tier2_inference.py when the candidate is an omni bundle (one
built with the workflow's `omni: true` input — i.e. vllm_omni is installed). It
boots the bundle's own `vllm-omni-server` on a small omni model with a
single-GPU deploy config and exercises the OpenAI-compatible chat endpoint, so
the qualification gate proves the omni multi-stage pipeline actually serves on
the GPU — not just that plain vLLM loads.

It emits a `tier2` fragment (same tier id as tier2_inference.py) so the existing
aggregate `--require-tiers tier0,tier1,tier2` promotion rule is unchanged; the
two scripts are mutually exclusive per run.

Checks:
  T2.1  omni server boot   — vllm-omni-server starts and /v1/models is ready (gating).
  T2.2  chat completion    — /v1/chat/completions returns non-empty text (gating).
  T2.3  streaming          — streamed chat yields content chunks + [DONE] (gating).

Usage:
    python3 tier2_omni.py --bundle-root ./vllm-install --gfx-target gfx1151 \
        --model Qwen/Qwen2.5-Omni-3B --output tier2-gfx1151.json
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

# Omni first-boot is slow: 3 stage engine cores + multimodal weight load +
# per-kernel Triton JIT for the gfx target. Allow generous headroom.
READY_TIMEOUT = 1800
REQUEST_TIMEOUT = 600

# Default deploy config shipped alongside this script (colocates all stages on
# device 0 for single-GPU qualification hardware).
DEFAULT_DEPLOY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "deploy", "qwen2_5_omni_1gpu.yaml"
)


def http_post(url, payload, timeout=REQUEST_TIMEOUT, stream=False):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return resp
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
        # Omni's frontend may not expose /health; /v1/models is the reliable
        # readiness signal once all stages have registered.
        if http_get(f"http://127.0.0.1:{port}/v1/models") == 200:
            return True, None
        if http_get(f"http://127.0.0.1:{port}/health") == 200:
            return True, None
        time.sleep(5)
    return False, f"not ready within {timeout}s"


def _tail(path, lines=60):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "\n".join(handle.read().splitlines()[-lines:])
    except OSError:
        return ""


class OmniServer:
    def __init__(self, root, model, deploy_config, port, log_path, max_model_len):
        self.root = root
        self.model = model
        self.deploy_config = deploy_config
        self.port = port
        self.log_path = log_path
        self.max_model_len = max_model_len
        self.proc = None

    def start(self):
        launcher = os.path.join(self.root, "bin", "vllm-omni-server")
        log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [
                launcher,
                "serve", self.model,
                "--omni",
                "--deploy-config", self.deploy_config,
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--max-model-len", str(self.max_model_len),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def sweep_orphans():
    """Last-resort kill of straggler omni server/stage processes (VRAM).

    Runs only AFTER the report is written. Anchored to the omni server's own
    process titles — the omni CLI entrypoint and its StageEngineCoreProc /
    EngineCore children + the multiprocessing resource_tracker — NOT bare
    "vllm" (which would also match this harness; see tier2_inference.py).
    """
    subprocess.run(
        [
            "pkill", "-9", "-if",
            r"vllm_omni\.entrypoints|vllm\.entrypoints|StageEngineCore|"
            r"VLLM::EngineCore|multiprocessing\.resource_tracker",
        ],
        check=False,
    )


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_chat(base, model):
    body = http_post(
        f"{base}/v1/chat/completions",
        {"model": model,
         "messages": [{"role": "user", "content": "In one sentence, what is a lemon?"}],
         "max_tokens": 64, "temperature": 0.0, "modalities": ["text"]},
    )
    msg = body.get("choices", [{}])[0].get("message", {})
    content = (msg.get("content") or "") + (msg.get("reasoning") or "")
    if content.strip():
        return (report.STATUS_PASS, None, {"sample": content[:120]})
    return (report.STATUS_FAIL, "empty chat content", {"body": body})


def check_streaming(base, model):
    resp = http_post(
        f"{base}/v1/chat/completions",
        {"model": model,
         "messages": [{"role": "user", "content": "Say hello."}],
         "max_tokens": 32, "temperature": 0.0, "modalities": ["text"],
         "stream": True},
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
        try:
            delta = json.loads(payload)["choices"][0].get("delta", {})
            if delta.get("content"):
                chunks += 1
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
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
    # dist-info dirs normalize to underscores: vllm_omni-<ver>.dist-info
    return ver("vllm"), torch_v, rocm_v, ver("vllm_omni")


def main():
    parser = argparse.ArgumentParser(description="Tier 2 omni functional inference")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--gfx-target", required=True)
    parser.add_argument("--channel", default=None, choices=[None, "stable", "nightly"])
    # Smallest validated omni model. AR (thinker/talker/code2wav) — no diffusion.
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--deploy-config", default=DEFAULT_DEPLOY)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--candidate-tag", default=None)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--logs-dir", default=".")
    parser.add_argument("--output", default="tier2-report.json")
    parser.add_argument("--ready-timeout", type=int, default=READY_TIMEOUT)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.bundle_root)
    vllm_v, torch_v, rocm_v, omni_v = detect_versions(root)
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

    if not omni_v:
        # Not an omni bundle — caller routed wrong. Record a clear fail so the
        # gate blocks rather than silently passing an omni-less bundle.
        tier.add("T2.1", "omni server boot", report.STATUS_FAIL,
                 error="vllm_omni not installed in bundle — not an omni build")
        tier.write(args.output)
        tier.print_summary()
        sys.exit(1 if args.fail_on_error else 0)

    if not os.path.exists(args.deploy_config):
        tier.add("T2.1", "omni server boot", report.STATUS_FAIL,
                 error=f"deploy config not found: {args.deploy_config}")
        tier.write(args.output)
        tier.print_summary()
        sys.exit(1 if args.fail_on_error else 0)

    os.makedirs(args.logs_dir, exist_ok=True)
    log_path = os.path.join(args.logs_dir, f"vllm-omni-server-{args.gfx_target}.log")
    base = f"http://127.0.0.1:{args.port}"
    server = OmniServer(root, args.model, args.deploy_config, args.port,
                        log_path, args.max_model_len)

    try:
        server.start()
        ready, err = wait_ready(args.port, server.proc, args.ready_timeout)
        if ready:
            tier.add("T2.1", "omni server boot", report.STATUS_PASS,
                     details={"model": args.model, "vllm_omni": omni_v})
        else:
            tier.add("T2.1", "omni server boot", report.STATUS_FAIL, error=err,
                     details={"log_tail": _tail(log_path)})

        if ready:
            tier.run("T2.2", "chat completion",
                     lambda: check_chat(base, args.model))
            tier.run("T2.3", "streaming",
                     lambda: check_streaming(base, args.model))
        else:
            for tid, name in [("T2.2", "chat completion"), ("T2.3", "streaming")]:
                tier.add(tid, name, report.STATUS_SKIP, error="server not ready")
    finally:
        server.stop()

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")

    sweep_orphans()

    if args.fail_on_error and tier.rollup_status() == report.STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
