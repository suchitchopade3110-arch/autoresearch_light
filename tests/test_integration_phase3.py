import subprocess
import os
import sys
import tempfile
import shutil

def test_full_loop_phase3_evolution():
    with tempfile.TemporaryDirectory() as temp_repo:
        for item in ["configs", "orchestrator", "eval", "sandbox", "vcs", "memory", "generation", "evolution",
                     "approval", "reporting", "observability"]:
            shutil.copytree(item, os.path.join(temp_repo, item))
        shutil.copy("config_schema.py", os.path.join(temp_repo, "config_schema.py"))

        with open(os.path.join(temp_repo, ".gitignore"), "w") as f:
            f.write("chroma_db/\n__pycache__/\n")

        with open(os.path.join(temp_repo, "candidate_script.py"), "w") as f:
            f.write("\n")

        subprocess.run(["git", "init"], cwd=temp_repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_repo, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_repo, check=False)
        subprocess.run(["git", "add", "."], cwd=temp_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_repo, check=True)

        config_path = os.path.join(temp_repo, "configs", "example.yaml")
        with open(config_path, "r") as f:
            config = f.read()
        config = config.replace("population_size: 5", "population_size: 2")
        config = config.replace("max_generations: 3", "max_generations: 2")
        # Unattended run, no dashboard operator present - see test_integration.py
        config = config.replace("  enabled: true", "  enabled: false")
        with open(config_path, "w") as f:
            f.write(config)

        env = os.environ.copy()
        env["PYTHONPATH"] = temp_repo

        goal = "Test Phase 3 Evolution"
        # sys.executable, not a bare "python" - see test_integration.py
        cmd = [sys.executable, "-m", "orchestrator.run", "--config", "configs/example.yaml", "--goal", goal, "--mode", "evolutionary"]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_repo, env=env)

        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0
        assert "Initializing Population" in result.stdout
        assert "Generation Report:" in result.stdout
        assert "=== Running Generation 2/2 ===" in result.stdout or "=== Running Generation 2/3 ===" in result.stdout

        assert os.path.exists(os.path.join(temp_repo, "evolution_report.jsonl"))
        with open(os.path.join(temp_repo, "evolution_report.jsonl"), "r") as f:
            lines = f.readlines()
            assert len(lines) >= 1