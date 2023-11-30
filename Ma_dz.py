import numpy as np
def sinusoidal_to_complex(amplitude, phase):
    complex_voltage = (amplitude/np.sqrt(2))* (np.cos(phase)+1j*np.sin(phase))
    return complex_voltage

z1=30j
z2=90 - 100j
z4=20+40j
z5=80j
z6=40+10j

e1=sinusoidal_to_complex(733.3, 3.831)
e2=sinusoidal_to_complex(949.2, 3.201)
e3=sinusoidal_to_complex(824.6, 1.030)

J=sinusoidal_to_complex(4.5, 3.463)


j1_1=J
j2_2 =  -3+4j
j3_3 = 5+2j

j1=j2_2
j2=j3_3
j3=j2_2-j1_1
j4=j1_1+j3_3
j5=j1_1
j6=j2_2+j3_3
arr_i=[j1,j2,j3,j4,j5,j6]


uj=e3+j4*z4+j5*z5
j11=j1.conjugate()
j22=j2.conjugate()
j33=j3.conjugate()
Jj=J.conjugate()
j1a=abs(j1)
j2a=abs(j2)
j4a=abs(j4)
j5a=abs(j5)
j6a= abs(j6)

g1=j1a**2
g2=j2a**2
g4=j4a**2
g5=j5a**2
g6=j6a**2

print ("Мощность источников:", e1*j11+e2*j22+e3*j33+Jj*uj)
print ("Мощность приемников:", g1*z1 + g2*z2 + g4*z4 + g5*z5 +g6*z6)