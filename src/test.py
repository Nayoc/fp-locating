from geopy.distance import geodesic

dist = geodesic((31.02846112, 121.4462418), (31.02846329, 121.4462407)).m
print(dist)
dist = geodesic((31.02846112, 121.4462418), (31.02846329, 121.4462407)).km
print(dist)
