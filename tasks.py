import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project"
PYTHON_VERSION = "3.11"

# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

@task
def git_push(ctx, message, branch='main'):
    """
    Perform git operations (add, commit, push) on a specified branch.
    If the branch does not exist locally or remotely, it will be created.
    After the operation, it will switch back to the original branch.
    
    Args:
        message (str): The commit message.
        branch (str): The branch to push to (default is 'main').
    """
    # Store the current branch to return to later
    current_branch = ctx.run("git rev-parse --abbrev-ref HEAD", hide=True).stdout.strip()

    # Fetch the latest branches from the remote (to ensure up-to-date remote references)
    ctx.run("git fetch origin", hide=True)
    
    ctx.run(f"git checkout {branch}")
    
    # Add changes
    ctx.run("git add .")
    
    # Commit changes
    ctx.run(f"git commit -m '{message}'")
    
    # Push to specified branch
    ctx.run(f"git push origin {branch}")
    
    ctx.run(f"git checkout {current_branch}")
