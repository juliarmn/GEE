#Sempre antes importar e inicializar
#Colocar as coordenadas (código intro.py)
#Função de máscara d'água:
def agua(imagem):
    pixel = imagem.select('qa')
    return pixel.bitwiseAnd(1 << 2).eq(0)
    
def nuvem(imagem):
    pixel = imagem.select('qa')
    return pixel.bitwiseAnd(1 << 3).eq(0) and (pixel.bitwiseAnd(1 << 5).eq(0)) and (pixel.bitwiseAnd(1 << 6).eq(0)) and (pixel.bitwiseAnd(1 << 7).eq(0))
