import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phiqonnect",
    version="1.0.1",
    author="Tsubasa Miyazaki",
    author_email="ts.miyazaki@scsk.jp",
    description="SCSK Quantum AI Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    python_requires='>=3.9',
)