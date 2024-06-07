import ee
ee.Authenticate()
ee.Initialize()
coordenadas =  "-68.43514524047421,  -58.65054484365196, 0.4165458387606975, -5.707281211922151"
x1, y1, x2, y2 = coordenadas.split(",")
geometria = geometry = ee.Geometry.Polygon (
                                            [
                                                [[float(x1), float(y2)],
                                                 [float(x1), float(y1)],
                                                 [float(x2), float(y2)],
                                                 [float(x2), float(y1)]
                                                ]
                                            ]
                                            )

longitude_centro = (float(y2) + float(y1)) / 2
latitude_centro = (float(x2) + float(x1)) / 2

data = "2017-01-01,2017-12-31"

inicio, fim = data.split(",")
print(inicio)
print(fim)

colecao = ee.ImageCollection("LANDSAT/LC08/C02/T1").filterBounds(geometria).filterDate(inicio, fim).filterMetadata('CLOUD_COVER', 'less_than', 100)

print(colecao.size().getInfo())
