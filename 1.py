import numpy as np
from math import atan, exp
w=1000
k1=np.sqrt((3*w)**2+1250**2)*exp(atan(3*w/1250))/((3*w)**2+1500**2)*exp(atan(3*w/1500))
k2=((9*w**2+1_875_000)**2-(750*w)**2)/((9*w**2)**2-2_250_000**2)
print (k1)
print (k2)