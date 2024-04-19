<p align="center">
<img src="https://raw.githubusercontent.com/bancaditalia/BeforeIT.jl/main/docs/logo/logo_black_text.png" width="500">
<sup><a href="#footnote-1">*</a></sup>
</p>

# Behavioural agent-based economic forecasting

Welcome to BeforeIT.jl, a package for **B**ehavioural agent-based **e**conomic **fore**casting,
from the **IT** research unit of the Bank of Italy.

BeforeIT.jl is a Julia-based implementation of the agent-based model presented in 
[_Economic forecasting with an agent-based model_](https://www.sciencedirect.com/science/article/pii/S0014292122001891),
the first ABM matching the forecasting performance of traditional economic tools.

With BeforeIT.jl, you can perform economic forecasting and explore different counterfactual scenarios.
Thanks to its modular design, the package is also a great starting point for anyone looking to extend its
capabilities or integrate it with other tools.

Developed in Julia, a language known for its efficiency, BeforeIT.jl is both fast and user-friendly,
making it accessible whether you’re an expert programmer or just starting out.

The package currently contains the original parametrisation for Austria, as well as a parametrisation for Italy.
Recalibrating the model on other nations is possible of course, but currently not easily supported.
So get in contact if you are interested!

## Julia installation

To run this software, you will need a working Julia installation on your machine.
If you don't have Julia installed already, simply follow the short instructions
available [here](https://julialang.org/downloads/).

## Installation

To be able to run the model, you can activate a new Julia environment in any folder from the terminal by typing

```
julia --project=.
```

Then, whithin the Julia environment, you can install BeforeIT.jl as

```julia
using Pkg
Pkg.add(url="git@github.com:bancaditalia/BeforeIT.jl.git")
```

You can ensure to have installed all dependencies via

```julia
Pkg.instantiate()
```

Now you should be able to run the the following code


```julia
using BeforeIT, Plots

parameters = BeforeIT.AUSTRIA2010Q1.parameters
initial_conditions = BeforeIT.AUSTRIA2010Q1.initial_conditions

T = 20
model = BeforeIT.initialise_model(parameters, initial_conditions, T)
data = BeforeIT.run_one_sim!(model)

plot(data.real_gdp)
```

This will simulate the model with the original Austrian parametrisation for 20 quarters.

In you want to run the script without opening a REPL, you can copy and paste the above lines into a file,
say `main.jl`, and run it directly from the terminal by typing

```
julia --project=. main.jl
```

## Disclaimer

This package is an outcome of a research project. All errors are those of
the authors. All views expressed are personal views, not those of Bank of Italy.

---

<p id="footnote-1">
* Credits to <a href="https://www.bankit.art/people/sara-corbo">Sara Corbo</a>  for the logo and to <a href="https://www.bankit.art/people/andrea-gentili">Andrea Gentili</a> for the name suggestion.
</p>