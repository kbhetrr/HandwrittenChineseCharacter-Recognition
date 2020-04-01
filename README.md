# Inception Module 기반의 한자 손글씨 분류 모델 제작
Handwritten Chinese Character Recognition with Inception Module

------

CASIA HWDB 1.1 데이터를 이용하여 3755개의 한자를 인식하는 모델을 제작.

클래스 수가 많아 기존의 CNN으로 모델을 구축하게 되면 파라미터 수가 급격히 늘어남.

GoogLeNet에서 처음 공개된 Inception Module을 이용하면 파라미터 수는 줄이고, 모델을 깊게 구축할 수 있음.

### 1. 학습 데이터
CASIA HWDB 1.1 Offline Database를 이용했습니다. (<a href="http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html">링크</a>)

|데이터 셋 명|클래스 수|데이터 개수|데이터 셋 크기|
|---------|--------  |----------|----|
| HWDB 1.1 | 3755 | **Train** : 897,758장 / **Test** : 223,991장 | **Train** : 1.9GB / **Test** : 0.4GB |
