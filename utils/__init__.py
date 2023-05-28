"""A module with utilities."""
import contextlib


@contextlib.contextmanager
def download_wordnet():
    """
    Load WordNet.

    :return: The WordNet.
    """
    import nltk

    nltk.download("wordnet")
    yield
