import os
import subprocess
import time
import uuid
from typing import Any, Dict, Optional


class SandboxExecutor:
    """
    Executes a script inside a Docker sandbox.

    SECURITY NOTE:
    Runs as a non-root user with no network access, a read-only root
    filesystem, and dropped capabilities, on top of the wall-clock
    timeout and CPU/memory limits. This raises the bar against a
    candidate script trying to exfiltrate data, persist state, or exceed
    its resource limits, but it is still a standard container, not a
    hardened micro-VM - it does NOT protect against a deliberate kernel
    exploit or container escape. Do not run untrusted malware here.
    """
    def __init__(self, config: Dict[str, Any], dataset_dir: Optional[str] = None):
        self.timeout = config.get('timeout_seconds', 10)
        self.cpu_limit = config.get('cpu_limit', '1.0')
        self.memory_limit = config.get('memory_limit', '512m')
        # train.jsonl/test.jsonl mounted read-only into every sandbox run if
        # set, so callers (the sequential loop and the concurrent
        # evolutionary scheduler alike) don't each need to know about
        # dataset wiring individually. The held-out labels file living
        # alongside them is NEVER mounted here - only the host-side eval
        # pipeline reads it, so a candidate can never read its own answer
        # key off disk. See eval/dataset.py and eval/pipeline.py.
        self.dataset_dir = os.path.abspath(dataset_dir) if dataset_dir else None
        self._build_image()

    def _build_image(self):
        subprocess.run(
            ["docker", "build", "-t", "ml-sandbox", "-f", "sandbox/Dockerfile", "sandbox/"],
            check=True,
            capture_output=True
        )

    def run_candidate(self, script_path: str, env_vars: Optional[Dict[str, str]] = None,
                       out_dir: Optional[str] = None) -> Dict[str, Any]:
        """Runs the given script inside the docker sandbox."""
        start_time = time.time()
        container_name = f"sandbox-{uuid.uuid4().hex[:8]}"

        cmd = [
            "docker", "run", "--rm",
            f"--name={container_name}",
            f"--cpus={self.cpu_limit}",
            f"--memory={self.memory_limit}",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "-v", f"{script_path}:/app/candidate_script.py:ro",
        ]

        run_env = dict(env_vars or {})
        if self.dataset_dir:
            train_path = os.path.join(self.dataset_dir, "train.jsonl")
            test_path = os.path.join(self.dataset_dir, "test.jsonl")
            cmd += ["-v", f"{train_path}:/app/data/train.jsonl:ro"]
            cmd += ["-v", f"{test_path}:/app/data/test.jsonl:ro"]
            run_env.setdefault("TRAIN_PATH", "/app/data/train.jsonl")
            run_env.setdefault("TEST_PATH", "/app/data/test.jsonl")

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            # A read-write bind mount coexists fine with --read-only on the
            # root filesystem - only this path is writable.
            cmd += ["-v", f"{os.path.abspath(out_dir)}:/app/out:rw"]

        for key, value in run_env.items():
            cmd += ["-e", f"{key}={value}"]

        cmd.append("ml-sandbox")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            execution_time = time.time() - start_time
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "timeout": False
            }
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            # explicitly stop the container to avoid orphaned processes
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            return {
                "exit_code": -1,
                "stdout": e.stdout.decode() if e.stdout else "",
                "stderr": e.stderr.decode() if e.stderr else f"Execution timed out after {self.timeout} seconds.",
                "execution_time": execution_time,
                "timeout": True
            }
