import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier('haarcascade\haarcascade-frontalface-default.xml')

reconhecedor = cv2.face.EigenFaceRecognizer_create()  # Carrega o método eigenface
reconhecedor.read('classificadores\classificadorEigenYaleComParametro.yml')  # Leitura do arquivo do treinamento

totalAcertos = 0
percentualAcerto = 0.0
totalConfianca = 0.0

caminhos = [os.path.join('yalefaces/teste', f) for f in os.listdir('yalefaces/teste')]
for caminhoImagem in caminhos:
    imagemFace = Image.open(caminhoImagem).convert('L')
    imagemFaceNP = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
    for (x, y, l, a) in facesDetectadas:
        idprevisto, confianca = reconhecedor.predict(imagemFaceNP)
        idatual = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        print(str(idatual) + " foi classificado como " + str(idprevisto) + " - " + str(confianca))
        if idprevisto == idatual:
            totalAcertos += 1
            totalConfianca += confianca
        cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 0, 255), 2)  # adiciona retangulo nas imagens
        cv2.imshow("Face", imagemFaceNP)  # exibe em uma janela "Face" as imagens
        cv2.waitKey(1000)
percentualAcerto = (totalAcertos / 30) * 100
totalConfianca = totalConfianca / totalAcertos
print('---Resultado da Avalicação Algoritmo Eigenface com Paramento---')
print("Percentual de acerto: " + str(percentualAcerto))
print("Total confiança: " + str(totalConfianca))
