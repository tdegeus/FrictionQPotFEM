# FrictionQPotFEM

[![Travis](https://travis-ci.com/tdegeus/FrictionQPotFEM.svg?branch=master)](https://travis-ci.com/tdegeus/FrictionQPotFEM)
[![Build status](https://ci.appveyor.com/api/projects/status/cx5ksr804rqidq0d?svg=true)](https://ci.appveyor.com/project/tdegeus/frictionqpotfem)

Friction simulations based on "GMatElastoPlasticQPot" and "GooseFEM"

# Testing

## Additional checks and balances

Additionally, consistency against earlier runs can be checked as follows.

### UniformSingleLayer2d - PNAS

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/PNAS-2019 ../examples/PNAS-2019_N=3\^2_id=000.h5
```

### UniformSingleLayer2d::System

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/UniformSingleLayer2d_System
cp ../examples/UniformSingleLayer2d_System.py .
python UniformSingleLayer2d_System.py ../examples/UniformSingleLayer2d_System.txt
```

### UniformSingleLayer2d::HybridSystem

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/UniformSingleLayer2d_HybridSystem
cp ../examples/UniformSingleLayer2d_HybridSystem.py .
python UniformSingleLayer2d_HybridSystem.py ../examples/UniformSingleLayer2d_HybridSystem.txt
```

### UniformSingleLayer2d - Load cycle

```none
cd build
cmake .. -DBUILD_TESTS=1 -DBUILD_EXAMPLES=1
make
./examples/LoadCycle
cp ../examples/LoadCycle.py .
python LoadCycle.py ../examples/LoadCycle.txt
```
