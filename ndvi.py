vermelho = 'B4'
infra = 'B5'

def ndvi(imagem):
    ndvi = imagem.expression('(infra - vermelho) / (infra + vermelho)', {'infra': imagem.select(infra), 'vermelho': imagem.select(vermelho)}).rename('ndvi')
    return imagem.addBands(ndvi)

colecao.map(ndvi)
imagem = colecao.median()
imagem_corte = imagem.clipToBoundsAndScale(geometry=geometria,scale=40)
imagem_corte = ndvi(imagem_corte)
print(imagem.bandNames().getInfo())

#Pegar imagem:
print(imagem_corte.select(['ndvi']).getDownloadUrl())

print()
print(imagem_corte.select(['ndvi']).getThumbUrl({'min':-1, 'max':1}))

#rgb:
print(imagem_corte.select(['B4','B3','B2']).getDownloadUrl())

print()
print(imagem_corte.select(['B4','B3','B2']).getThumbUrl({'min':0, 'max':3000}))
