---
title: "[논문 리뷰] De-identifying health records by means of active learning"
categories:
  - AL
tags:
  - AL
  - NLP
  - Medical Domain
---


***
- Date : 2012
- Dataset : Clinical Text Dataset(EMR)
- 목적 : De-identifying Clinical Records
- 모델 : Random Forest
- 쿼리 기법 : uncertainty가 가장 높은 데이터, 가장 낮은 데이터, 그리고 랜덤하게 선택하는 기법
***



# Abstract

해당 논문은 의료 데이터의 단어(토큰)들을 PHI(protected health information)의 <span style="color:red">8개 라벨</span>로 분류하는 태스크를 다루고 있다. 모델은 <span style="color:red">**Random Forest**</span>, 그리고 query 기법은 <span style="color:red">**uncertainty가 가장 높은 데이터, 가장 낮은 데이터, 그리고 랜덤하게 선택하는 기법**</span> 세 가지를 비교하였다. 그 결과 uncertainty가 낮을 때, 랜덤, uncertainty 가 높을 때 순으로 성능이 높았다. 

# Introduction

**의료 전자 문서에 포함된 정보를 확인**하기 위해서는 **de-identification** 과정이 필요하다. 이를 자동화하기 위한 머신러닝 기법들이 존재한다. 주요 정보를 포함한 단어들에 태그를 부여하는 역할을 한다. 하지만 머신러닝 기법들도 학습을 위해 많은 양의 라벨이 포함된 데이터가 필요하기 때문에 이러한 단점을 완화하기 위해 AL 이 등장했다. 가장 대표적인 데이터 query 방법이 uncertainty가 가장 낮은 데이터를 선택하는 방법이다.

# Methods

데이터 : 스웨덴어로 쓰인 health record 100개(pain, orthopedy, oral, maxillofacial surgery, diet 5개 분야에 대한 문서)

라벨 class : age, part of date, full date, 성, 이름, health care unit, location, phone number, non-phi value

사용한 feature : 주로 토큰 정보에 대한 것으로, 영숫자 여부, 숫자 여부, pos tag, 대문자로 시작하는지 여부, 앞뒤 토큰의 pos tag, 토큰 길이, 앞뒤 토큰의 토큰 길이 총 13가지 정보를 feature로 활용하였다. 

**사용한 쿼리 방법 : uncertain, most certain, random**

모델 : 100개의 tree 로 구성된 **random forest**

초기 데이터 10000개 토큰과 query된 10000개의 토큰을 이용하여 98820개의 토큰을 테스트하였다. 라벨 분포는 non-phi가 96.74%로 대다수를 차지하고 있었다. 

평가 metric : 총 10개의 metric을 이용하였고, precision, recall, f1-score 등 binary 한 태스크를 대상으로 하는 metric에 대해서는 각 class 별로 해당 클래스로 예측하였을 때 T, 해당 클래스로 예측하지 않았을 때 F를 부여하여 계산하였고, 이들 간의 micro, macro 평균을 계산하였다. 

# Results

![image](https://user-images.githubusercontent.com/69342517/215096410-21e90bdf-379b-401b-80de-b05ffbec7552.png){: width="30%" height="30%" .align-center}
초기 모델에 비해 정확도는 많이 높아지지 않았지만, test 데이터의 양을 고려하면 정확하게 예측한 데이터의 양이 훨씬 많다는 점을 알 수 있다. 또한, 세 가지 query 방법 모두 초기 모델의 AUC가 0.5 인 것에 비해 좋은 결과를 보였다. Error도 36% 가량 감소하였다. 

# Discussion

이러한 결과가 나온 이유는 크게 두 가지로 설명될 수 있다. 먼저, 사용한 feature 개수가 태그를 부여하기에 부족했다. 또한, <span style="color:red">**class 분포가 편향**</span>되어 있다. 따라서 후속 연구에서는 class 분포 편향의 영향을 줄일 수 있는 방법과 유용한 feature를 활용하는 방법으로 진행될 수 있다.