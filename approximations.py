import numpy as np
from scipy.special import jn  # Importing the j0 function

# Define constants and data arrays
R = 4 #number of resources
out = 10 #length of output samples
phase_shift = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9])
P_1 = np.array([0.19672402619617100000, 0.17944352000247800000, 0.16146550626352600000,
                0.11949720031367500000, 0.06226986498344590000, 0.02995015788875120000]) #enter P_1 values


def sinc(x):
    if x == 0:
        return 1.0
    else:
        return np.sin(x) / x


# Compute y1, y2, y3 for each pair of values from phase_shift and P_1
y1_sinc_values = []
y2_sinc_values = []
y3_sinc_values = []

for ps, p in zip(phase_shift, P_1):
    y1_sinc_value = (p ** np.abs(R)) * np.abs(sinc(ps)) ** out + p * (1 - np.abs(sinc(ps)) ** out) #for beta=1
    y2_sinc_value = (p ** np.abs(R)) * np.abs(sinc(ps)) ** (out/2) + p * (1 - np.abs(sinc(ps)) ** (out/2)) #for beta=1/2
    y3_sinc_value = (p ** np.abs(R)) * np.abs(sinc(ps)) ** (out*2) + p * (1 - np.abs(sinc(ps)) ** (out*2)) #for beta =2
    
    y1_sinc_values.append(y1_sinc_value)
    y2_sinc_values.append(y2_sinc_value)
    y3_sinc_values.append(y3_sinc_value)

print("y1_sinc_values: ",y1_sinc_values)
print("y2_sinc_values:", y2_sinc_values)
print("y3_sinc_values:", y3_sinc_values)

# Compute y1, y2, y3 for each pair of values from phase_shift and P_1
y1_j0_values = []
y2_j0_values = []
y3_j0_values = []

for ps, p in zip(phase_shift, P_1):
    y_j0_1 = (p ** abs(R)) * np.abs(jn(0, ps))**out + p * (1 - np.abs(jn(0, ps))**out) #beta=1
    y_j0_2 = (p ** abs(R)) * np.abs(jn(0, ps))**(out/2) + p * (1 - np.abs(jn(0, ps))**(out/2)) #beta = 1/2
    y_j0_3 = (p ** abs(R)) * np.abs(jn(0, ps))**(out*2) + p * (1 - np.abs(jn(0, ps))**(out*2)) #beta=2
    
    y1_j0_values.append(y_j0_1)
    y2_j0_values.append(y_j0_2)
    y3_j0_values.append(y_j0_3)

print("y1_j0_values:",y1_j0_values)
print("y2_j0_values:",y2_j0_values)
#print("y3_j0_values:",y2_j0_values)
