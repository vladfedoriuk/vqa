"""Configuration utilities."""
import contextlib


@contextlib.contextmanager
def load_env_config():
    """Load environment variables."""
    from config import env

    yield env
