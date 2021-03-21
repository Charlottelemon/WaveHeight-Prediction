import netCDF4 as nc
import math

fname = 'wave.bohai.yellow2020.nc'
f = nc.Dataset(fname)
msldata = 'msl'
u10data = 'u10'
v10data = 'v10'
swhdata = 'swh'

msll = f[msldata]
u10 = f[u10data]
v10 = f[v10data]
swhh = f[swhdata]


msl = []
uv10 = []
swh = []

#print(len(msl))
for i in range(1464):
    for j in range(12):
        msl.append(msll[i][j])
        uv10.append(math.sqrt(u10[i][j]**2 + v10[i][j]**2))
        swh.append(swhh[i][j])
