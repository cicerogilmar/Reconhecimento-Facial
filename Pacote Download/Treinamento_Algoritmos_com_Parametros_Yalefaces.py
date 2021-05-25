import cv2
import os
import numpy as np
from PIL import Image  # biblioteca para fazer carregamento de imagens que estão em disco

eigenface = cv2.face.EigenFaceRecognizer_create(120, 1500)
fisherface = cv2.face.FisherFaceRecognizer_create(4, 1800)
lbph = cv2.face.LBPHFaceRecognizer_create(1, 9, 9, 9, 1800)


def getImagemComId():
    caminhos = [os.path.join('yalefaces/treinamento', f) for f in os.listdir('yalefaces/treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        ids.append(id)
        faces.append(imagemNP)

    return np.array(ids), faces


ids, faces = getImagemComId()

print("Treinando com Parametro...")
eigenface.train(faces, ids)
eigenface.write('classificadores\classificadorEigenYaleComParametro.yml')

fisherface.train(faces, ids)
fisherface.write('classificadores\classificadorFisherYaleComParametro.yml')

lbph.train(faces, ids)
lbph.write('classificadores\classificadorLBPHYaleComParametro.yml')

print("Treinamento com Paramentro Realizado")
