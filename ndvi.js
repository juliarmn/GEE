//Imports:
var L8: ImageCollection LANDSAT/LC08/C02/T1_TOA
var parametro: B5 from -1

//ee.Image : funções  para cálculos
var img = L8.filterBounds(ee.Geometry.Point(-4, 59.6)).filterMetadata('CLOUD_COVER', 'less_than', 1).first();
var vermelho = img.select('B4');
var infraVermelho = img.select('B5');

//Cálculo do NDVI(Normalize Difference Vegetation Index)-> quantidade de verde em um lugar
//Diferença entre a faixa vermelha e infravermelha dividida pela soma das duas
//O Google Earth tem uma função própria para isso, mas vamos calcular assim:
var NDVI = infraVermelho.subtract(vermelho).divide(infraVermelho.add(vermelho));
Map.addLayer(NDVI, parametro)
//Maior NDVI indica vegetação verde


//Cálculo NDVI usando a função:
//Imports:
var L8: ImageCollection LANDSAT/LC08/C02/T1_TOA
var parametro: B5 from -1

//ee.Image : funções  para cálculos
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
//Pode usar essa expressão tbm:
var NDVI = img.expression('(b(\'B5\') - b(\'B5\'))/(b(\'B5\') + b(\'B5\'))')