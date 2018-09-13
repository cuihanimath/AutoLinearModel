import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoLinearModel",
    version="0.0.1",
    author="Han Cui & Jason (Zishuo) Li",
    author_email="hc2813@columbia.edu & zl2528@columbia.edu",
    description="Auto linear regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cuihanimath/AutoLinearModel",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: columbia",
        "Operating System :: OS Independent",
    ),
)