import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

## Get the long description from the README file
# long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="phone-similarity",
    version="0.0.1",  # pre-release
    description="A library dedicated to providing phonological distance and similarity metrics.",
    # long_description=long_description,  # Optional
    # long_description_content_type="text/markdown",  # Optional (see note above)
    # url="https://github.com/pypa/sampleproject",  # Optional
    author="Kleber Noel",  # Optional
    # author_email="author@example.com",  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are: 3 - Alpha; 4 - Beta; 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="phone, phonological, phonetic, similarity, ipa, phonetics, distance, bitarray",  # Optional
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=[
        "transformers~=4.48",
        "bitarray",
        "langcodes[data]",
        "numpy~=1.26",
        "torch>=2,<=3",
    ],
)
