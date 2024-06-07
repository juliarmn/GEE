var L8: ImageCollection LANDSAT/LC08/C02/T1_TOA
var parametro: B5 from -1
var img = L8.filterBounds(ee.Geometry.Point(-4, 59.6)).filterMetadata('CLOUD_COVER', 'less_than', 1).first();
var vermelho = img.select('B4');
var infraVermelho = img.select('B5');

var NDVI = infraVermelho.subtract(vermelho).divide(infraVermelho.add(vermelho));
Map.addLayer(NDVI, parametro)
var L8: ImageCollection LANDSAT/LC08/C02/T1_TOA
var parametro: B5 from -1

var img = L8.filterBounds(ee.Geometry.Point(-4, 59.6)).filterMetadata('CLOUD_COVER', 'less_than', 1).first();
var vermelho = img.select('B4');
var infraVermelho = img.select('B5');
var NDVI = img.expression('(infraVermelho - vermelho)/(infraVermelho + vermelho)',
                            {
                            'infraVermelho': infraVermelho,
                            'vermelho': vermelho
                            }
                           )
Map.addLayer(NDVI, parametro, 'NDVI')
var NDVI = img.expression('(b(\'B5\') - b(\'B5\'))/(b(\'B5\') + b(\'B5\'))')
