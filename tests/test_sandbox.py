import json
import os
import tempfile
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
