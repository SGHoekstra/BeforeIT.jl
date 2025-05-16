#from sbi4abm.sbi.analysis.conditional_density import (
#    conditional_corrcoeff,
#    eval_conditional_density,
##)
#from sbi4abm.sbi.analysis.plot import conditional_pairplot, pairplot
#from sbi4abm.sbi.analysis.sensitivity_analysis import ActiveSubspace

# sbi/analysis/__init__.py
# Corrected content:
from .conditional_density import (  # Relative import
    conditional_corrcoeff,
    eval_conditional_density,
)
from .plot import conditional_pairplot, pairplot  # Relative import
from .sensitivity_analysis import ActiveSubspace  # Relative import