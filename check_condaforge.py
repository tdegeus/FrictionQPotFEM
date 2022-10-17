import re

import conda_envfile
import requests


def include_recipe(name):
    for regex in ["cxx-compiler", "pybind11-abi", "pip"]:
        if re.match(regex, name):
            return False
    return True


def include_requirement(name):
    for regex in ["catch2", "doxygen", "graphviz", "fmt"]:
        if re.match(regex, name):
            return False
    return True


response = requests.get(
    "https://raw.githubusercontent.com/conda-forge/frictionqpotfem-feedstock/main/recipe/meta.yaml"
)
meta = response.text
env = conda_envfile.condaforge_dependencies(meta)
recipe = []
for i in env:
    if include_recipe(i.name):
        recipe.append(i)
recipe = conda_envfile.unique(*recipe)

env = conda_envfile.parse_file("environment.yaml")["dependencies"]
requirements = []
for i in env:
    if include_requirement(i.name):
        requirements.append(i)
requirements = conda_envfile.unique(*requirements)

if not conda_envfile.contains(requirements, recipe):
    conda_envfile.print_diff(requirements, recipe)
