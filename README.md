# Inception Module 기반의 한자 손글씨 분류 모델 제작
Handwritten Chinese Character Recognition with Inception Module

------

CASIA HWDB 1.1 데이터를 이용하여 3755개의 한자를 인식하는 모델을 제작.
http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

클래스 수가 많아 기존의 CNN 모델을 사용하면 잘 분류하지 못함.

GoogLeNet에서 처음 공개된 Inception Module을 이용하면 파라미터 수는 줄이고, 모델을 깊게 구축할 수 있음.
