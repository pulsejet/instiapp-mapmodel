"""Map model for InstiApp GPS location.

Copyright 2019 Varun Patil <radialapps@gmail.com>

This work is licensed under the terms of the MIT license.  
For a copy, see <https://opensource.org/licenses/MIT>.
"""

import math
import numpy as np
from PIL import Image

# Path and constants of map files
MAP_FILE = 'map.jpg'
BLUE_MARKER_FILE = 'marker_blue_s.png'
RED_MARKER_FILE = 'marker_red_s.png'
MARKER_SIZE = 200, 200

# Raw data points
locs = [
    (19.133691, 72.916984, 4189, 1655, "Intersection of infinities"),
    (19.133013, 72.917822, 4277, 1863, "VMCC intersection"),
    (19.123948, 72.911225, 781, 2776,  "Lakeside gate"),
    (19.128356, 72.919225, 3830, 3000, "YP gate"),
    (19.137600, 72.915069, 4335, 1017, "H15 intersection"),
    (19.136127, 72.910546, 3207, 949,  "H4 intersection"),
    (19.130776, 72.917189, 3714, 2209, "Shirucafe"),
    (19.130811, 72.914857, 3091, 1964, "Hospi road"),
    (19.125455, 72.916304, 2262, 3236, "Main Gate"),
    (19.133674, 72.911850, 3004, 1272, "New gymkhana road"),
    (19.135893, 72.906964, 2538, 768,  "H6"),
    (19.129592, 72.915441, 3041, 2245, "H10 Intersection"),
    (19.129700, 72.919007, 4025, 2596, "KV-YP Intersection"),
    (19.133009, 72.913769, 3338, 1540, "Swimming pool road - main road intersection"),
    (19.134272, 72.910148, 2805, 1099, "CH intersection"),
    (19.136017, 72.914713, 3993, 1189, "SAC intersection"),
    (19.136282, 72.918019, 4787, 1373, "QIP intersection"),
    (19.134384, 72.918330, 4611, 1637, "QIP intersection lower"),
    (19.135437, 72.906146, 2291, 771,  "H12 intersection"),
    (19.136024, 72.912494, 3540, 1068, "H2 road"),
    (19.129959, 72.910325, 1945, 1769, "Boat house"),
    (19.128436, 72.914645, 2605, 2393, "Jalvihar lake intersection lower"),
    (19.125301, 72.913422, 1547, 2803, "Jayantia intersection"),
    (19.131663, 72.915277, 3383, 1867, "Convo som intersection"),
    (19.136237, 72.916970, 4550, 1306, "21-Type1 intersection"),
    (19.137621, 72.917248, 4768, 1150, "Vidya niwas intersection"),
    (19.132018, 72.919453, 4493, 2149, "Aero border intersection"),
    (19.128468, 72.917833, 3401, 2767, "C7 intersection"),
    (19.132843, 72.915994, 3783, 1727, "Main building intersection"),
    (19.134365, 72.915378, 3933, 1447, "Library intersection"),
]

# Unzip
all_coords = list(map(list, zip(*locs)))
X = np.array(all_coords[0])
Y = np.array(all_coords[1])
Z = np.array(all_coords[2])
Zy = np.array(all_coords[3])

Xn = X[0]
Yn = Y[0]
Zn = Z[0]
Zyn = Zy[0]

print()
print('Origin - Xn=%s, Yn=%s, Zn=%s, Zyn=%s' % (Xn, Yn, Zn, Zyn))
print()

factor = 1000

X = (X - Xn) * factor
Y = (Y - Yn) * factor
Z = Z - Zn
Zy = Zy - Zyn

# Process
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
Zy = Zy.flatten()

# Fit
K = [X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]
A = np.array(K).T

coeff, r, rank, s = np.linalg.lstsq(A, Z, rcond=None)
coeffy, ry, ranky, sy = np.linalg.lstsq(A, Zy, rcond=None)

# Print our coefficients
print()
print(', '.join(list(str(x) for x in coeff)))
print()
print(', '.join(list(str(x) for x in coeffy)))
print()

# Predictor
def c(x, y):
    x = (x - Xn) * factor
    y = (y - Yn) * factor
    A = coeff
    pixel_x = Zn + A[0] + A[1]*x + A[2]*y + A[3]*x**2 + A[4]*x**2*y + A[5]*x**2*y**2 + A[6]*y**2 + A[7]*x*y**2 + A[8]*x*y
    A = coeffy
    pixel_y = Zyn + A[0] + A[1]*x + A[2]*y + A[3]*x**2 + A[4]*x**2*y + A[5]*x**2*y**2 + A[6]*y**2 + A[7]*x*y**2 + A[8]*x*y
    return pixel_x, pixel_y

loss = 0
def addloss(x, y, pixel_x, pixel_y, name):
    global loss
    
    # Add and print loss
    x, y = c(x, y)
    closs = np.round(math.sqrt((x - pixel_x)**2 + (y - pixel_y)**2), 2)
    print(name, closs)
    loss += closs

print("TRAINING LOSS")
for x in locs:
    addloss(*x)
print("TOTAL", np.round(loss, 2))
print()

# Do some validation
loss = 0
print("VALIDATION LOSS")
valid = [
    (19.132018, 72.918015, 4167, 2055, "VMCC lower"),
    (19.134360, 72.915061, 3841, 1429, "Library"),
    (19.133483, 72.912373, 3103, 1361, "H11"),
    (19.138056, 72.916654, 4685, 1047, "Aravali?"),
    (19.128764, 72.915857, 2947, 2432, "H10"),
]
for x in valid:
    addloss(*x)
print("TOTAL", np.round(loss, 2))
print()

# ===================================================================
# Make beautiful assets
# ===================================================================

# Open and load map images
image = Image.open(open(MAP_FILE, 'rb'))
blue_marker = Image.open(open(BLUE_MARKER_FILE, 'rb'))
blue_marker.thumbnail(MARKER_SIZE, Image.ANTIALIAS)
red_marker = Image.open(open(RED_MARKER_FILE, 'rb'))
red_marker.thumbnail(MARKER_SIZE, Image.ANTIALIAS)
marker_width, marker_height = blue_marker.size
conv_mark = lambda coords: (coords[0] - (marker_width // 2), coords[1] - (marker_height))

for x in valid:
    image.paste(red_marker, conv_mark((x[2], x[3])), red_marker)
for x in locs:
    image.paste(blue_marker, conv_mark((x[2], x[3])), blue_marker)

image.save('modelmap.jpg', 'JPEG', quality=90, optimize=True, progressive=True)
