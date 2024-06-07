import ee
import pandas as pd
import numpy as np
ee.Initialize()

def water_mask(image):
    qa = image.select('pixel_qa')
    return qa.bitwiseAnd(1 << 2).eq(0)

def cloud_mask(image):
    qa = image.select('pixel_qa')
    return qa.bitwiseAnd(1 << 3).eq(0) and (qa.bitwiseAnd(1 << 5).eq(0)) and (qa.bitwiseAnd(1 << 6).eq(0)) and (qa.bitwiseAnd(1 << 7).eq(0))

def apply_masks(image):
    blank = ee.Image.constant(0).selfMask()
    agua = blank.updateMask(water_mask(image).Not()).rename('agua')
    nuvem = blank.updateMask(cloud_mask(image).Not()).rename('nuvem')
    sem_nuvem = blank.updateMask(cloud_mask(image)).rename('sem_nuvem')
    ndvi = image.expression('(nir - red) / (nir + red)',{'nir':image.select('B5'),'red':image.select('B4')}).rename('ndvi')
    return image.addBands([ndvi,agua,nuvem,sem_nuvem])

def band_mask(image, band, origin, dest):
    image_with_masks = image.select(origin).updateMask(image.select(band)).rename(dest) 
    image_with_masks = ee.Image.constant(0).selfMask()
    image_with_masks = image_with_masks.blend(image_with_masks).rename(dest)
    return image.addBands([image_with_masks])


# coordenadas = "-66.53801472648439,-3.503806214013736,-66.270222978437516,-3.7281869567509"
# x1,y1,x2,y2 = coordenadas.split(",")

#Uso do geojson.io:
geometria = geometry = ee.Geometry.Polygon(
        [[[-68.43514524047421, 0.4165458387606975],
          [-68.43514524047421,-5.707281211922151],
          [-58.65054484365196, -5.707281211922151],
          [-58.65054484365196, 0.4165458387606975],
          [-68.43514524047421, 0.4165458387606975]]])

datas = "2014-10-6,2014-11-14"

inicio,fim = datas.split(",")

colection_image = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio,fim).filterMetadata('CLOUD_COVER','less_than', 30)
colection_image = colection_image.map(apply_masks)
imagem = colection_image.median()
print(imagem.bandNames().getInfo())
imagem = band_mask(imagem, 'agua', 'ndvi', 'ndvi_agua')
imagem = band_mask(imagem, 'nuvem', 'ndvi', 'ndvi_nuvem')
imagem = band_mask(imagem, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
imagem = band_mask(imagem, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')
imagem_corte = imagem.clipToBoundsAndScale(geometry=geometria,scale=30)


def extract_coordinate_data(image, geometry, bands):
    image = image.addBands(ee.Image.pixelLonLat())
    coordinates = image.select(['longitude', 'latitude']+bands).reduceRegion(reducer=ee.Reducer.toList(),geometry=geometria,scale=30,bestEffort=True)
    band_value = []

    for band in bands:
        band_value.append(np.array(ee.List(coordinates.get(band)).getInfo()).astype(float))
    return np.array(ee.List(coordinates.get('latitude')).getInfo()).astype(float), np.array(ee.List(coordinates.get('longitude')).getInfo()).astype(float), band_value

longitudes, latitudes, index = extract_coordinate_data(imagem_corte, geometria, ['ndvi_agua_sem_nuvem'])

print(len(latitudes))
print(len(longitudes))
print(len(index[0]))
