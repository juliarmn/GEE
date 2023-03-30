import ee
import pandas as pd
import numpy as np
import PIL
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

ee.Initialize() 

def water_mask(image):
    qa = image.select('pixel_qa')
    return qa.bitwiseAnd(1 << 2).eq(0)

def cloud_mask(image):
    qa = image.select('pixel_qa')
    return qa.bitwiseAnd(1 << 3).eq(0) and (qa.bitwiseAnd(1 << 5).eq(0)) and (qa.bitwiseAnd(1 << 6).eq(0)) and (qa.bitwiseAnd(1 << 7).eq(0))

# função para aplicar as máscaras
def apply_masks(image):
    # image vazia:
    blank = ee.Image.constant(0).selfMask()
    # máscara de água
    agua = blank.updateMask(water_mask(image).Not()).rename('agua')
    # image apenas com nuvens: se não tiver nuvem, ela ficará branca
    nuvem = blank.updateMask(cloud_mask(image).Not()).rename('nuvem')
    # Remover as nuvens:
    sem_nuvem = blank.updateMask(cloud_mask(image)).rename('sem_nuvem')
    # NDVI
    ndvi = image.expression('(nir - red) / (nir + red)',{'nir':image.select('B5'),'red':image.select('B4')}).rename('ndvi')
    # Adiciona as bandas
    return image.addBands([ndvi,agua,nuvem,sem_nuvem])

# Máscara em banda específica
def band_mask(image, band, origin, dest):
    #Banda de origem será direcionada para a de destino
    image_with_masks = image.select(origin).updateMask(image.select(band)).rename(dest)
    #image em branco
    image_with_masks = ee.Image.constant(0).selfMask()
    image_with_masks = image_with_masks.blend(image_with_masks).rename(dest)
    #Retornar a image com a nova banda nomeada com a string da banda_destino
    return image.addBands([image_with_masks])


#Função modificada para corrigir o problema do bitwise, descartando anomalias presentes em corpos de água
#IMPORTANTE
def apply_mask_modified(image, image_mask, banda_origem, band_destino): 
    image_masks = image.select(banda_origem).updateMask(image_mask).rename(band_destino)
    #image em branco para receber a banda e enviar para a banda de destino
    image_masks = ee.Image.constant(0).selfMask().blend(image_masks).rename(band_destino)
    return image.addBands([image_mask])

#Uso do geojson.io:
geometria = geometry = ee.Geometry.Polygon(
        [[[-68.43514524047421, 0.4165458387606975],
          [-68.43514524047421,-5.707281211922151],
          [-58.65054484365196, -5.707281211922151],
          [-58.65054484365196, 0.4165458387606975],
          [-68.43514524047421, 0.4165458387606975]]])

datas = "2014-10-6,2014-11-14"

#unpacking
inicio,fim = datas.split(",")

colection_image = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio,fim).filterMetadata('CLOUD_COVER','less_than', 30)
colection_image = colection_image.map(apply_masks)
image = colection_image.median()

# Cria a banda 'ndvi_agua'
image = image.addBands(image.select('ndvi').rename('ndvi_agua'))

# Cria as outras bandas usando a função band_mask()
image = band_mask(image, 'agua', 'ndvi', 'ndvi_agua')
image = band_mask(image, 'nuvem', 'ndvi', 'ndvi_nuvem')
image = band_mask(image, 'sem_nuvem', 'ndvi', 'ndvi_sem_nuvem')
image = band_mask(image, 'agua', 'ndvi_sem_nuvem', 'ndvi_agua_sem_nuvem')

image_cut = image.clipToBoundsAndScale(geometry=geometria,scale=30)


def extract_coordinate_data(image, geometry, bands):
    image = image.addBands(ee.Image.pixelLonLat())
    coordinates = image.select(['longitude', 'latitude']+bands).reduceRegion(reducer=ee.Reducer.toList(),geometry=geometry,scale=30,bestEffort=True)

    #Cria um numpy array
    band_value = []
    #Extrai o valor de band 1 a 1:
    for band in bands:
        #Coloca o valir na nossa lista numpy
        #ee.List: transformar os pixeis em lista
        #getInfo(): Extrair os pixels
        band_value.append(np.array(ee.List(coordinates.get(band)).getInfo()).astype(float))
    return np.array(ee.List(coordinates.get('latitude')).getInfo()).astype(float), np.array(ee.List(coordinates.get('longitude')).getInfo()).astype(float), band_value

#_______________________________________________________________________________________________________________________________________________________________________
datas_treinamento = "2014-10-13,2014-10-14"
datas_classificacao = "2014-01-30,2014-01-31"
inicio_treinamento, fim_treinamento = datas_treinamento.split(",")
inicio_classificacao, fim_classificacao = datas_classificacao.split(",")
#Consultando a coleção pela área e data, image para extração do corpo d'água
water_mask_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate('2018-01-01','2020-01-01').filterMetadata('CLOUD_COVER','less_than', 10)
water_mask_collection = water_mask_collection.map(apply_masks)
water_mask_water = water_mask_collection.median()
#image de Treinamento
training_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio_treinamento,fim_treinamento).filterMetadata('CLOUD_COVER','less_than', 10)
training_collection = training_collection.map(apply_masks)
training = training_collection.median()
#image de Classificação
classification_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR').filterBounds(geometria).filterDate(inicio_classificacao,fim_classificacao).filterMetadata('CLOUD_COVER','less_than', 10)
classification_collection = classification_collection.map(apply_masks)
classification_img = classification_collection.median()

#Correção de bitwise
mascara_agua = ee.Image(0).blend(ee.Image.constant(0).selfMask().updateMask(water_mask_water.select('agua').gt(0))).eq(99999)
training = apply_mask_modified(training, mascara_agua, 'ndvi', 'ndvi_agua')
training_cut = training.clipToBoundsAndScale(geometry=geometria,scale=30)
longitudes, latitudes, index = extract_coordinate_data(training_cut, geometria, ['ndvi_agua'])

dataFrame_training = pd.DataFrame({
    'latitude': latitudes,
    'longitude': longitudes,
    'ndvi': index[0]
}, columns=['latitude', 'longitude', 'ndvi'])

#Cria colunas com valores booleanos, retorna true se for diferente de 99999
#O resultado é um novo DataFrame que contém apenas as linhas onde o valor da coluna "ndvi" é diferente de 99999 
#Remove os dados inválidos (dummies)
dataFrame_training = dataFrame_training[dataFrame_training['ndvi'] != 99999]
print(dataFrame_training.head())

