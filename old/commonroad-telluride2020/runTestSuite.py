import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('unitTests')

from test_derivatives import test_derivatives
from test_zeroInitialVelocity import test_zeroInitialVelocity

#run tests
res = test_derivatives()
res = test_zeroInitialVelocity()

