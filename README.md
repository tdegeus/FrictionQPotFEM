# FrictionQPotFEM

[![CI](https://github.com/tdegeus/FrictionQPotFEM/workflows/CI/badge.svg)](https://github.com/tdegeus/FrictionQPotFEM/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/FrictionQPotFEM/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/FrictionQPotFEM)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/frictionqpotfem.svg)](https://anaconda.org/conda-forge/frictionqpotfem)

Friction simulations based on "GMatElastoPlasticQPot" and "GooseFEM"

# Testing

# Additional checks and balances

Additionally, consistency against earlier runs can be checked as follows.

## UniformSingleLayer2d - PNAS

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/PNAS-2019 ../examples/PNAS-2019_N=3\^2_id=000.h5
```

## UniformSingleLayer2d::System

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/UniformSingleLayer2d_System
cp ../examples/UniformSingleLayer2d_System.py .
python UniformSingleLayer2d_System.py ../examples/UniformSingleLayer2d_System.txt
```

## UniformSingleLayer2d::HybridSystem

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/UniformSingleLayer2d_HybridSystem
cp ../examples/UniformSingleLayer2d_HybridSystem.py .
python UniformSingleLayer2d_HybridSystem.py ../examples/UniformSingleLayer2d_HybridSystem.txt
```

## UniformSingleLayer2d - Load cycle

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/LoadCycle
cp ../examples/LoadCycle.py .
python LoadCycle.py ../examples/LoadCycle.txt
```

# Generating the docs

## Basic

```
cd docs
doxygen
```

## With dependencies

For example using conda

```
cd docs
( cat Doxyfile ; echo "INPUT += ${CONDA_PREFIX}/include/GooseFEM" ; echo "INPUT += ${CONDA_PREFIX}/include/GMatElastoPlasticQPot" ) | doxygen -
```

# Installation

# C++ headers

## Using conda

```bash
conda install -c conda-forge frictionpotfem
```

## From source

```bash
# Download FrictionQPotFEM
git checkout https://github.com/tdegeus/FrictionPotFEM.git
cd FrictionQPotFEM

# Install headers and CMake support
cmake -Bbuild .
cd build
make install
```

# Python module

## Using conda

```bash
conda install -c conda-forge python-frictionpotfem
```

Note that *xsimd* and hardware optimisations are **not enabled**.
To enable them you have to compile on your system, as is discussed next.

## From source

>   You need *xtensor*, *xtensor-python* and optionally *xsimd* as prerequisites.
>   Additionally, Python needs to know how to find them.
>   The easiest is to use *conda* to get the prerequisites:
>
>   ```bash
>   conda install -c conda-forge xtensor-python
>   conda install -c conda-forge xsimd
>   ```
>
>   If you then compile and install with the same environment
>   you should be good to go.
>   Otherwise, a bit of manual labour might be needed to
>   treat the dependencies.

```bash
# Download FrictionQPotFEM
git checkout https://github.com/tdegeus/FrictionPotFEM.git
cd FrictionQPotFEM

# Only if you want to use hardware optization:
export CMAKE_ARGS="-DUSE_SIMD=1"

# Compile and install the Python module
# (-vv can be omitted as is controls just the verbosity)
python -m pip install . -vv
```

# Change-log

## v0.16.6

*   initEventDriven: Removing assertion that makes no sense (c depends on the realization)

## v0.16.5

*   Bugfix initEventDriven: adding missing references (#127)
*   eventDriven: add "yield_element" option

## v0.16.4

*   eventDriven: override with additional update to UniformMultiLayerLeverDrive2d (#126)
*   eventDriven: choosing lowest scale factor (rather than shortest distance in eq. strain space) (#125)
*   Adding assertions (#124)

## v0.16.3

*   eventDriven: fixing incorrect check (#123)

## v0.16.2

*   UniformMultiLayerIndividualDrive2d: adding convenience function (deprecated integrated functions)

## v0.16.1

*   [docs] Fixing typo
*   Removing deprecated eventDriven code. The API is kept in deprecated mode, but "dry_run == true" it removed immediately.
*   Identifying error in deprecated eventDriven protocol
*   UniformSingleLayer2d: adding "typical_plastic_h", "typical_plastic_dV", "affineSimpleShear", "affineSimpleShearCentered"
*   Adding "timeSteps_boundcheck" and "flowSteps_boundcheck"

## v0.16.0

*   Add "timeSteps" and "flowSteps" (#119)
*   Adding event driven protocol to all systems (#118)

## v0.15.1

*   Improving version_dependencies(). Various minor clean-ups (#115)

## v0.15.0

*   Adding "reset_epsy"
*   Adding "minimise_boundcheck".
*   Bugfix "minimise" return the number of iterations (not the iteration index)
*   [Python] Removing unnecessary imports
*   Formatting updates (#113)

## v0.14.2

*   [clang-format] Adding formatting suggestion
*   [pre-commit] Applying pre-commit (#112)
*   Version list: sorting alphabetically
*   [Python] Minor bugfix: wrong import

## v0.14.1

*   Fixing missing version string.

## v0.14.0

*   [Python] Switching to scikit-build
*   Renaming "test" -> "tests"
*   Updating GooseFEM::Iterate -> minor stopping criterion change (#109)
*   Fixing 'bug': computing forces only when possible (#108)
*   HybridSystem: dealing with empty elastic or plastic definitions (#106)
*   Improving assertion messages (#106)
*   Minor bugfixes in UniformSingleLayer2d::LocalTrigger (#100)
*   [CI] Using assertions
*   [tests] Minor bugfixes (#106)
*   [setup.py] Improve verbosity
*   [CMake] Add CMake options USE_ASSERT, USE_DEBUG, USE_SIMD (#102)
*   [CMake] Enable Python (deprecation) warnings

## v0.13.0

Multi-layer: overhaul, adding drive with a lever (#95, #96)

*   Simplifying tests
*   Adding "UniformMultiLayerLeverDrive2d"

*   Changing protected functions "UniformMultiLayerIndividualDrive2d":
    *   Splitting "computeForceDrive" -> "computeLayerUbarActive"/"computeForceFromTargetUbar"
    *   Adding "updated_target_ubar"

*   Changing API "UniformMultiLayerIndividualDrive2d":

    *   Renaming "layerSetTargetUbarAndDistribute" -> "layerSetUbar"
    *   Renaming "setDriveStiffness" -> "layerSetDriveStiffness"
    *   Adding "layerSetTargetActive" to activate driving springs
    *   Adding "layerTargetActive"
    *   Adding "layerTagetUbar_addAffineSimpleShear" that only affects the driving frame
    *   "addAffineSimpleShear" (only affects body)
    *   "layerSetTargetUbar" (activation of springs now in "layerSetTargetActive")

*   Internal renaming
*   Code-style update

## v0.12.3

*   Avoiding setuptools_scm dependency if SETUPTOOLS_SCM_PRETEND_VERSION is defined
*   Multi-layer: add shear without distributing it

## v0.12.2

*   setup.py: use CMAKE_ARGS environment variable; removing pyxtensor dependency

## v0.12.1

*   timeStepsUntilEvent: allow using a maximum number of iterations

## v0.12.0

*   Bugfix & API extension: initializing target ubar and returning its value
*   API change: renaming "fdrivespring" -> "layerFdrive"
*   API change: renaming "layerSetUbar" -> "layerSetTargetUbar" and "layerSetDistributeUbar" -> "layerSetTargetUbarAndDistribute"
*   API change: renaming "nlayers" -> "nlayer"
*   API change: returning isplastic for all layers
*   Minor efficiency update: avoiding temporary (#89)

## v0.11.5

*   `addSimpleShearToFixedStress`: making assertion on elastic step optional in
    `addElasticSimpleShearToFixedStress` (#88)
*   Using GMatElastoPlasticQPot::version() (#87)

## v0.11.4

*   Removing myargsort workaround (#86)
*   Bugfix: bug manifesting itself only in the Python API (#85)
*   Adding function to set time
*   Adding code to get the number of layers
*   Doxystyle update

## v0.11.3

*   Function to run time-steps until the next plastic event (#82)
*   Wider application of xt::pytensor (#81)
*   Removing deprecated GooseFEM functions from tests (#80)

## v0.11.2

*   Using xtensor-python (#79)
*   Multi-layers: skipping of computations where possible
*   Multi-layers: return ubar
*   Multi-layers: Allow asymmetric drive spring
*   Multi-layers: Get driving force per layer

## v0.11.1

*   Multi-layers: apply simple shear drive
*   Multi-layers: distribute drive displacement

## v0.11.0

*   Integrating Python API in CMake (#73)
*   Minor update multi-layer example. Temporarily switch off trigger test on GCC (#72)
*   Adding multi-layered simulations (#69)
*   Updating doxygen-awesome
*   Minor CMake updates

## v0.10.0

*   Branching common methods from UniformSingleLayer2d to Generic2d
    (UniformSingleLayer2d now only has one class based on the HybridSystem).
*   Making returned references explicit.
*   Adding deprecation warnings Energy() to Python API.

## v0.9.4

*   Class members: pass by reference (instead of copy); works also in Python API

## v0.9.1

*   Python API: forcing copy of certain objects (#62)
*   Using CMake for Doxygen (#61, #62)
*   Adding dependencies to docs

## v0.9.0

*   Adding convenience method "Energy"
*   Removing namespace aliases
*   Getting mass and damping matrix

## v0.8.0

*   Updating versioning. Python API: auto-overloading (#57)
*   Using setuptools_scm to manage version (#56)
*   Documenting version information. Adding eigen to version string.
*   Renaming "versionInfo" -> "version_dependencies"
*   Adding fdamp and setV/setA
*   [CI] Switching to GCC 8 for the moment (#51)
*   Examples: modifying to API change, removing namespace abbreviation.
*   Removing GoosFEM alias

## v0.7.0

*   Adding addAffineSimpleShearCentered (#41)

## v0.6.0

*   [CI] Removing travis and appveyor
*   Adding possibility to restart (#40)
*   Clang/Windows switching-off xtensor::optimize (#39)
*   Getting plastic strain (#38)
*   Reformatting CI: using clang on Windows
*   Compute sign of displacement perturbation
*   Add affine shear step
*   Adding "currentYield*" offset overload
*   LocalTrigger: adding option to evaluate only small slice; making energy relative to the
    volume of a plastic element (not to that of the system); adding simple-shear search
*   Splitting tests in several sources, enable Windows CI
*   Trigger: most output
*   Triggering: optimisations & minimal search (#34)
    -   Trigger: Optimizations
    -   Adding minimal search
    -   Changing call of parent constructor
*   Deprecated local energy barriers (#33)
*   Implementation triggering of smallest energy barrier (#31)
    -   API Change: Removing "init" functions, using constructors directly.
        Adding stiffness matrix.
    -   Adding triggering protocol
*    Exploring energy barrier upon trigger (#30)
*    Adding plastic_ElementYieldBarrierForSimpleShear (#28)
*    Exploring energy landscape to local simple shear perturbation (#24)
*    Temporarily excluding Windows from GitHub CI
*    Adding GitHub CI
*    Adding Python API
*    Amplifying trigger
