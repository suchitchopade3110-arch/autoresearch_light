import subprocess
import os
import sys
import tempfile
import shutil

def test_full_loop_phase2():
    # Create a temporary directory to act as the repository
    with tempfile.TemporaryDirectory() as temp_repo:
        # Copy the project files to the temp directory
        for item in ["configs", "orchestrator", "eval", "sandbox", "vcs", "memory", "generation", "approval",
                     "reporting", "observability"]:
            shutil.copytree(item, os.path.join(temp_repo, item))
        shutil.copy("config_schema.py", os.path.join(temp_repo, "config_schema.py"))

        # create gitignore to prevent chroma_db from being tracked and failing git checkouts
        with open(os.path.join(temp_repo, ".gitignore"), "w") as f:
            f.write("chroma_db/\n__pycache__/\n")

        with open(os.path.join(temp_repo, "candidate_script.py"), "w") as f:
            f.write("\n")

        # Unattended run, no dashboard operator present - see test_integration.py
        config_path = os.path.join(temp_repo, "configs", "example.yaml")
        with open(config_path) as f:
            config_text = f.read()
        config_text = config_text.replace("  enabled: true", "  enabled: false")
        with open(config_path, "w") as f:
            f.write(config_text)

        # Initialize a new git repository
        subprocess.run(["git", "init"], cwd=temp_repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_repo, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_repo, check=False)
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_repo, check=True)

        # Run the loop in the temporary directory with a specific goal
        goal = "Test Phase 2 Mock"
        # sys.executable, not a bare "python" - see test_integration.py
        cmd = [sys.executable, "-m", "orchestrator.run", "--config", "configs/example.yaml", "--goal", goal]

        env = os.environ.copy()
        env["PYTHONPATH"] = temp_repo
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_repo, env=env)

        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0
        assert "Generated diff:" in result.stdout
        assert "Candidate passed all evaluation stages." in result.stdout
        assert "Merging." in result.stdout

        # Verify the record ended up in the DB
        # We can write a quick python script to run inside the temp repo to check the DB
        check_db_script = f"""
from memory.db import ExperimentDB
db = ExperimentDB()
results = db.retrieve_experiments("{goal}", k=1)
assert len(results) > 0, "No records found in DB"
assert results[0]['outcome'] == 'success', f"Expected success outcome, got {{results[0]['outcome']}}"
print("DB check passed")
"""
        with open(os.path.join(temp_repo, "check_db.py"), "w") as f:
            f.write(check_db_script)

        db_result = subprocess.run([sys.executable, "check_db.py"], capture_output=True, text=True, cwd=temp_repo, env=env)
        assert db_result.returncode == 0
        assert "DB check passed" in db_result.stdout

def test_failure_retrieval_influences_prompt():
    with tempfile.TemporaryDirectory() as temp_repo:
        for item in ["configs", "orchestrator", "eval", "sandbox", "vcs", "memory", "generation"]:
            shutil.copytree(item, os.path.join(temp_repo, item))

        # create gitignore to prevent chroma_db from being tracked
        with open(os.path.join(temp_repo, ".gitignore"), "w") as f:
            f.write("chroma_db/\n__pycache__/\n")

        # Create candidate_script.py
        with open(os.path.join(temp_repo, "candidate_script.py"), "w") as f:
            f.write("\n")

        subprocess.run(["git", "init"], cwd=temp_repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_repo, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_repo, check=False)
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_repo, check=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = temp_repo

        # Inject a failure directly into DB
        inject_script = """
from memory.db import ExperimentDB
db = ExperimentDB()
db.store_experiment(
    hypothesis="Test Retrieve Failure",
    diff="dummy diff",
    rationale="cause failure",
    metrics={},
    outcome="failure",
    failure_reason="KNOWN_INJECTED_FAILURE_REASON"
)
"""
        with open(os.path.join(temp_repo, "inject.py"), "w") as f:
            f.write(inject_script)
        subprocess.run([sys.executable, "inject.py"], cwd=temp_repo, env=env, check=True)

        # Now run a script that builds a prompt and verifies it contains the failure
        check_prompt_script = """
import yaml
from memory.db import ExperimentDB
from generation.prompt_builder import PromptBuilder

with open("configs/example.yaml", 'r') as f:
    config = yaml.safe_load(f)

db = ExperimentDB()
pb = PromptBuilder(db, config.get('generation', {}))
prompt = pb.build_prompt("Test Retrieve Failure")

assert "KNOWN_INJECTED_FAILURE_REASON" in prompt, "Failure reason not found in prompt"
print("Prompt builder successfully included the past failure.")
"""
        with open(os.path.join(temp_repo, "check_prompt.py"), "w") as f:
            f.write(check_prompt_script)

        res = subprocess.run([sys.executable, "check_prompt.py"], capture_output=True, text=True, cwd=temp_repo, env=env)
        assert res.returncode == 0
        assert "Prompt builder successfully included the past failure" in res.stdout
