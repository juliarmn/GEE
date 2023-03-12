var L8: ImageCollection LANDSAT/LC08/C02/T1
var region:Polygon, 4 vertices
var Cor: B4, B3 and B2

var filteredRegion = L8.filterBounds(region);
var filteredMeta = filteredRegion.filterMetadata('CLOUD_COVER', 'less_than', 15)
var filteredDate = filteredMeta.filterDate('2021-01-01', '2023-02-10')
var medianImage = filteredDate.median()
Map.addLayer(medianImage, Cor, "Layer");