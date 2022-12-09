from pathlib import Path

from setuptools_scm import get_version
from skbuild import setup

project_name = "FrictionQPotFEM"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=project_name,
    description="Friction model based on GooseFEM and FrictionQPotFEM.",
    long_description=long_description,
    version=get_version(),
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    url=f"https://github.com/tdegeus/{project_name}",
    packages=[f"{project_name}"],
    package_dir={"": "python"},
    cmake_install_dir=f"python/{project_name}",
    cmake_minimum_required_version="3.13",
)
