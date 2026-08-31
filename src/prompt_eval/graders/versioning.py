import subprocess


def get_prompt_version() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "prompts/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    return f"{commit}-dirty" if dirty else commit
