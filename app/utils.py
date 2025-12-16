import hashlib


def make_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
