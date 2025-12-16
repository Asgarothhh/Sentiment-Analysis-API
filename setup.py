import pathlib

import pkg_resources
from setuptools import find_packages, setup


def requirements(filepath: str):
    """Парсинг зависимостей из requirements.txt"""
    with pathlib.Path(filepath).open() as requirements_txt:
        return [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]


setup(
    name="sentiment_app",
    version="0.1.0",
    description="Sentiment Analysis API на FastAPI с Redis и PyTorch",
    author="Asgaroth",
    python_requires=">=3.10",
    packages=find_packages(include=["app", "ml", "app.*", "ml.*"], exclude=["tests"]),
    install_requires=requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: FastAPI",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
