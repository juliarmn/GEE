import ee
ee.Authenticate()
ee.Initialize()
coordenadas =  "-3.10719,  -60.0261, -3.9000, -64.5000"
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
#Queremos as coordenadas centrais da área:
longitude_centro = (float(y2) + float(y1)) / 2
latitude_centro = (float(x2) + float(x1)) / 2
#Definir por data as imagens que queremos:
data = "2017-01-01,2017-12-31"
#Unpacking:
inicio, fim = data.split(",")
print(inicio)
print(fim)
#Filtrar coleções:
#1 - Pegar a coleção desejada, vamos usar o landsat-08
colecao = ee.ImageCollection("LANDSAT/LC08/C02/T1").filterBounds(geometria).filterDate(inicio, fim).filterMetadata('CLOUD_COVER', 'less_than', 100)
#Ver o total de imagens disponíveis:
print(colecao.size().getInfo())
