import setuptools


def _requires_from_file(filename: str):
    return open(filename).read().splitlines()


setuptools.setup(
    name="deepext_with_lightning",
    version="0.1.0",
    install_requires=_requires_from_file("requirements.txt")
)
