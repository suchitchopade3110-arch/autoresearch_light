import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from sandbox.executor import SandboxExecutor

@pytest.fixture
def sandbox():
    # Setup Sandbox with 10 second timeout for normal tests
    config = {
        'timeout_seconds': 10,
        'cpu_limit': "0.5",
        'memory_limit': "256m"
    }
    return SandboxExecutor(config)

@pytest.fixture
def sandbox_short_timeout():
    # Setup Sandbox with 1 second timeout
    config = {
        'timeout_seconds': 1,
        'cpu_limit': "0.5",
        'memory_limit': "256m"
    }
    return SandboxExecutor(config)

def test_sandbox_successful_execution(sandbox):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("print('Hello from test')\n")
        script_path = f.name

    try:
        # absolute path for Docker volume
        abs_path = os.path.abspath(script_path)
        result = sandbox.run_candidate(abs_path)

        assert result['exit_code'] == 0
        assert not result['timeout']
        assert "Hello from test" in result['stdout']
    finally:
        os.remove(script_path)

def test_sandbox_timeout_enforcement(sandbox_short_timeout):
    # A script that sleeps for 5 seconds (sandbox timeout is 1 sec)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import time\ntime.sleep(5)\nprint('Done')\n")
        script_path = f.name

    try:
        abs_path = os.path.abspath(script_path)
        result = sandbox_short_timeout.run_candidate(abs_path)

        assert result['exit_code'] == -1
        assert result['timeout'] is True
        assert "Done" not in result['stdout']
    finally:
        os.remove(script_path)

def test_sandbox_failure(sandbox):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("raise ValueError('Test error')\n")
        script_path = f.name

    try:
        abs_path = os.path.abspath(script_path)
        result = sandbox.run_candidate(abs_path)

        assert result['exit_code'] != 0
        assert not result['timeout']
        assert "Test error" in result['stderr']
    finally:
        os.remove(script_path)

def test_sandbox_train_test_mount():
    with tempfile.TemporaryDirectory() as dataset_dir:
        with open(os.path.join(dataset_dir, "train.jsonl"), "w") as f:
            f.write(json.dumps({"x1": 1.0, "x2": 0.5, "label": 1}) + "\n")
        with open(os.path.join(dataset_dir, "test.jsonl"), "w") as f:
            f.write(json.dumps({"id": 0, "x1": 0.2, "x2": -0.1}) + "\n")
        # truth.json lives alongside train/test but is never mounted - see
        # sandbox/executor.py and test_reward_hacking.py.
        with open(os.path.join(dataset_dir, "truth.json"), "w") as f:
            f.write(json.dumps({"0": 1}))

        config = {'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m"}
        sandbox_with_dataset = SandboxExecutor(config, dataset_dir=dataset_dir)

        script_content = (
            "import os\n"
            "print('SUBSET_PERCENTAGE=' + os.environ.get('SUBSET_PERCENTAGE', 'missing'))\n"
            "print('TRAIN_PATH=' + os.environ.get('TRAIN_PATH', 'missing'))\n"
            "print('TEST_PATH=' + os.environ.get('TEST_PATH', 'missing'))\n"
            "print(open(os.environ['TRAIN_PATH']).read())\n"
            "print(open(os.environ['TEST_PATH']).read())\n"
            "print('TRUTH_VISIBLE=' + str(os.path.exists('/app/data/truth.json')))\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            abs_script = os.path.abspath(script_path)
            result = sandbox_with_dataset.run_candidate(abs_script, env_vars={"SUBSET_PERCENTAGE": "5"})

            assert result['exit_code'] == 0
            assert "SUBSET_PERCENTAGE=5" in result['stdout']
            assert "TRAIN_PATH=/app/data/train.jsonl" in result['stdout']
            assert "TEST_PATH=/app/data/test.jsonl" in result['stdout']
            assert '"x1": 1.0' in result['stdout']
            assert '"id": 0' in result['stdout']
            assert "TRUTH_VISIBLE=False" in result['stdout']
        finally:
            os.remove(script_path)

def test_truth_json_unreachable_by_any_path_inside_the_sandbox():
    """
    Hard invariant (see eval/pipeline.py's top-of-file comment). Stronger
    than test_sandbox_train_test_mount's single-path check: rather than
    confirming truth.json is absent at the one path a candidate might
    guess, this walks the ENTIRE container filesystem looking for a file
    literally named truth.json anywhere - proof that no path construction
    trick (relative, absolute, a symlink, an env-var-derived path) could
    ever reach it, because it simply isn't present anywhere the container
    can see. truth.json lives on disk right next to train.jsonl/test.jsonl
    (see eval/dataset.py); this is only true because sandbox/executor.py
    mounts individual files, never the whole directory.
    """
    with tempfile.TemporaryDirectory() as dataset_dir:
        with open(os.path.join(dataset_dir, "train.jsonl"), "w") as f:
            f.write(json.dumps({"x1": 1.0, "x2": 0.5, "label": 1}) + "\n")
        with open(os.path.join(dataset_dir, "test.jsonl"), "w") as f:
            f.write(json.dumps({"id": 0, "x1": 0.2, "x2": -0.1}) + "\n")
        with open(os.path.join(dataset_dir, "truth.json"), "w") as f:
            f.write(json.dumps({"0": 1}))

        config = {'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m"}
        sandbox_with_dataset = SandboxExecutor(config, dataset_dir=dataset_dir)

        script_content = (
            "import os\n"
            "found = []\n"
            "for root, dirs, files in os.walk('/'):\n"
            "    if 'truth.json' in files:\n"
            "        found.append(os.path.join(root, 'truth.json'))\n"
            "print('FOUND=' + str(found))\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            abs_script = os.path.abspath(script_path)
            result = sandbox_with_dataset.run_candidate(abs_script)
            assert result['exit_code'] == 0, result['stderr']
            assert "FOUND=[]" in result['stdout'], result['stdout']
        finally:
            os.remove(script_path)


def test_sandbox_out_dir_is_writable_and_survives_the_container(sandbox):
    with tempfile.TemporaryDirectory() as out_dir:
        script_content = (
            "import json\n"
            "with open('/app/out/predictions.jsonl', 'w') as f:\n"
            "    f.write(json.dumps({'id': 0, 'pred': 1}) + '\\n')\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            abs_path = os.path.abspath(script_path)
            result = sandbox.run_candidate(abs_path, out_dir=out_dir)
            assert result['exit_code'] == 0

            pred_path = os.path.join(out_dir, "predictions.jsonl")
            assert os.path.exists(pred_path)
            with open(pred_path) as f:
                assert json.loads(f.read().strip()) == {"id": 0, "pred": 1}
        finally:
            os.remove(script_path)

def test_sandbox_has_ml_libraries(sandbox):
    """Wave 2 acceptance: a candidate whose premise is ML research needs more than pure stdlib."""
    script_content = (
        "import numpy, pandas, sklearn\n"
        "print('ML_LIBS_OK', numpy.__version__, pandas.__version__, sklearn.__version__)\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        abs_path = os.path.abspath(script_path)
        result = sandbox.run_candidate(abs_path)
        assert result['exit_code'] == 0, result['stderr']
        assert "ML_LIBS_OK" in result['stdout']
    finally:
        os.remove(script_path)

def test_sandbox_has_no_network_access(sandbox):
    script_content = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    print('CONNECTED')\n"
        "except OSError:\n"
        "    print('NO_NETWORK')\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        abs_path = os.path.abspath(script_path)
        result = sandbox.run_candidate(abs_path)
        assert "NO_NETWORK" in result['stdout']
    finally:
        os.remove(script_path)


def test_docker_run_command_includes_resource_hardening_flags():
    """
    Wave 4 acceptance: CPU/memory limits alone don't bound process count or
    open file descriptors (a fork bomb or fd-exhaustion loop isn't CPU/
    memory-bound), and an unbounded /tmp tmpfs can exhaust host RAM since
    tmpfs is RAM-backed. Doesn't need a real docker daemon - mocks
    subprocess.run and inspects the constructed command directly.
    """
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        script_path = f.name

    try:
        with patch("sandbox.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            config = {
                'timeout_seconds': 10,
                'cpu_limit': "0.5",
                'memory_limit': "256m",
                'pids_limit': 64,
                'ulimit_nofile': 512,
                'tmpfs_size_mb': 32,
            }
            executor = SandboxExecutor(config)
            executor.run_candidate(script_path)

            run_call = next(c for c in mock_run.call_args_list if c.args[0][:2] == ["docker", "run"])
            cmd = run_call.args[0]

            assert "--pids-limit=64" in cmd
            assert "--ulimit" in cmd
            assert "nofile=512" in cmd
            assert "--tmpfs" in cmd
            assert "/tmp:size=32m" in cmd
    finally:
        os.remove(script_path)


def test_gpu_access_is_absent_by_default():
    """
    Priority 1 acceptance: GPU passthrough is an isolation trade-off the
    operator must opt into explicitly - the default config must never grant
    it. Doesn't need a real docker daemon or GPU.
    """
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        script_path = f.name

    try:
        with patch("sandbox.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            executor = SandboxExecutor({'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m"})
            executor.run_candidate(script_path)

            run_call = next(c for c in mock_run.call_args_list if c.args[0][:2] == ["docker", "run"])
            cmd = run_call.args[0]

            assert "--gpus" not in cmd
    finally:
        os.remove(script_path)


def test_gpu_access_is_granted_only_when_explicitly_configured():
    """A configured sandbox.gpus value is passed straight through to `docker run --gpus`."""
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        script_path = f.name

    try:
        with patch("sandbox.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            executor = SandboxExecutor({'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m", 'gpus': "all"})
            executor.run_candidate(script_path)

            run_call = next(c for c in mock_run.call_args_list if c.args[0][:2] == ["docker", "run"])
            cmd = run_call.args[0]

            assert "--gpus" in cmd
            assert "all" in cmd
    finally:
        os.remove(script_path)


def test_run_candidate_widens_permissions_before_mounting():
    """
    Wave 4 acceptance (CI regression): a bind mount carries the HOST file's
    real permission bits into the container - a script created with a
    restrictive mode (e.g. 0600, which tempfile.NamedTemporaryFile uses by
    default) or an out_dir left at the default 0755 from os.makedirs is
    unreadable/unwritable to the sandbox's non-root UID on native Linux
    Docker. This was silently masked on Docker Desktop for Windows/macOS
    (whose VM-backed bind mounts don't enforce host permission bits the
    same way), which is how it first shipped - only surfaced once this ran
    on a native Linux CI runner. Doesn't need a real docker daemon.
    """
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        script_path = f.name
    os.chmod(script_path, 0o600)

    with tempfile.TemporaryDirectory() as parent:
        out_dir = os.path.join(parent, "out")

        try:
            with patch("sandbox.executor.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                executor = SandboxExecutor({'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m"})
                executor.run_candidate(script_path, out_dir=out_dir)

            # Checked as "at least as permissive as", not exact equality -
            # os.chmod on Windows has no POSIX-style granularity (it only
            # toggles a read-only attribute), so os.stat there reports a
            # fixed 0o666/0o444 regardless of the exact mode passed in.
            # The bug this guards against is Linux/Docker-specific in the
            # first place (see the docstring above); this assertion only
            # needs to confirm the chmod call happened and didn't leave the
            # restrictive 0600/0755 bits in place, on any platform.
            assert os.stat(script_path).st_mode & 0o004, "script must be world-readable"
            assert os.stat(out_dir).st_mode & 0o002, "out_dir must be world-writable"
        finally:
            os.remove(script_path)
