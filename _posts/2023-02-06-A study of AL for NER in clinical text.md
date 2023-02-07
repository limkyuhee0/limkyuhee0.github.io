---
title: "[논문 리뷰] A study of AL for NER in clinical text(2015) - ing"
date : 2023-02-06
use_math: true
categories:
  - AL
tags:
  - AL
  - NLP
  - Medical Domain
  - NER
---



# Abstract

- Objective : Active Learning(AL)을 이용한 clinical NER 태스크 수행
- Methods
    - Dataset : 2010 i2b2/VA NLP challenge(349 documents, 20423 sentences)
    - AL 알고리즘 : uncertainty, diversity, baseline sampling vs passive learning
    - Measure : ALC(area under learning curve; AUC 랑 같은 의미인듯), F-measure vs sentences, F-measure vs words
    - 결과 : Uncertainty 기반의 방법은 42% annotation 감소, Diversity 기반의 방법은 7% annotation 감소로 비교적 좋지 않은 성능을 보였다.

# Introduction

- Objective : **identification of clinical concepts** or **clinical NER**
    - extract clinically important entities from clinical text(EMR에서 의미 있는 concept 찾기)
- 기존 Methods
    - ML-based model - SVM, CRF 가 가장 많이 사용된다. medication related entity extraction 태스크에서 최고 성능은 F-measure 85.23% 정도이다. 하지만 모델의 학습을 위해 라벨이 부여된 다량의 데이터셋이 필요하고, 이를 위한 annotation 과정에 비용과 시간이 많이 소모된다.
    - AL - 주로 pool 기반의 모델이 활용되고 word sense disambiguation, text classification, information extraction 등의 태스크를 다룬다. 해당 연구에서는 assertion classfication, supervised word sense disambiguation in MEDLINE, phenotyping tasks for EMR data 등의 의료 도메인 NLP 태스크를 대상으로 AL 을 적용한다.
- **NER** 태스크 - 다른 태스크들과 달리 **sequential labeling** 태스크이다. 따라서 NER 태스크에서 AL은 정보량이 많은 “**sequence**”를 찾는 것이 목표이다.
    - MMR 논문([https://limkyuhee0.github.io/al/Semi-supervised-active-learning-for-modeling-medical-concepts-from-free-text/](https://limkyuhee0.github.io/al/Semi-supervised-active-learning-for-modeling-medical-concepts-from-free-text/)) : uncertainty(N best sequence entropy) + diversity(각 단어를 3차원 벡터로 표현해서 문장간 유사도를 고려) - 혼합된 AL은 좋은 결과를 보였지만 diversity 만으로는 좋은 결과를 보이지 않았다.
    - Settles(2008) 논문 : 6개의 corpora에 대해 17가지 방법을 비교하였다. baseline(random sampling, long sentence sampling) vs uncertainty based, query-by-committee based, information density, fisher information, expected gradient length - 전반적으로 baseline에 비해 AL 기법이 좋은 결과를 보였다.
    
    ⇒ 기존 연구의 단점 : 모든 문장의 labeling cost 가 동일하다고 가정하였다. 하지만 실제 문장의 길이에 따라 labeling cost 가 다르다. 따라서 informativeness 를 측정할 다른 방법을 사용해야 한다.
    
- Online Learning : 주로 대량의 데이터나 실시간 데이터에 적합한 방법이다. 가능한 모든 데이터에 대해 학습하고자 하는 Batch 기반의 모델과 다르게 대량의 데이터에서 랜덤으로 샘플링을 해서 빠르게 모델을 생성한다. 대표적으로 OASIS는 실시간(stream) 데이터로부터 online learning과 semi-supervised learning을 이용하여 모델을 훈련하고 AL을 이용하여 인풋 데이터를 필터링하고자 한다.
- Pre-annotation : 모델을 이용하여 미리 라벨을 부여하는 작업을 수행하는 방법이다. 미리 라벨링된 데이터는 편향을 줄 수 있다는 단점이 있지만 dictionary-based pre-annotation은 편향을 줄일 수 있다는 연구가 있고, 편향이 다소 있더라도 비용과 시간을 절약할 수 있다는 장점이 있다.

해당 연구에서는 clinical NER 태스크를 다루는 데이터셋에 **13가지 AL** 기법(6개의 기존 기법과 7개의 새로운 기법)을 비교하였고, 실제 상황에서는 문장의 길이, 즉 **단어의 개수에 따라 annotation cost 가 달라진다는 점을 고려**하여 **단어의 개수에 따른 결과**도 비교하였다. 

# 2. Methods

### 2.1. Dataset & 2.2. ML-based NER

349 clinical documents, 20423 sentences

모든 문장에 대해 annotation을 하였다. 라벨은 **problem, treatment, test** 그리고 이를 구성하는 구문에 대한 **BIO 라벨과 함께** 부여하였다. 즉, **“B-problem”,”B-treatment”,”B-test”,“I-problem”,”I-treatment”,”I-test”,”O” 총 7개의 라벨**을 갖는다. 이 중 80%를 pool로 사용하고 20%를 test 용도로 활용하였다. 모델은 **CRF**를 활용하였다.

### 2.3. AL experimental framework

해당 연구에서는 다음과 같은 과정으로 AL 을 적용하였다.

1. Initial model generation : 초기 모델 학습을 위한 데이터 선정에는 random sampling 과 longest sentence sampling 중 후자를 사용하였다. 후자가 learning curve 초기에 더 좋은 결과를 보이기 때문에 선택하였지만 어떤 방법을 선택하던지 최종 결과에 유의한 영향을 주지는 않는다. 
2. Querying : 라벨이 부여되지 않은 set에서 querying 알고리즘에 따라 라벨을 부여할 데이터를 선정한다. 이 때 uncertainty 기반의 알고리즘과 같이 업데이트된 CRF 모델을 필요로 하기도 한다. 각 iteration마다 선정할 데이터는 iteration $i$ 당 $2^{(i+2)}$개만큼 선정하였다. 
3. Training & Iteration : Pool 안의 모든 데이터가 query 될 때까지 Query와 CRF 모델 학습 과정을 반복한다. 

평가지표는 F-measure와 annotated set의 엔티티 개수, 단어 개수 등을 활용한다. 사용한 AL 알고리즘은 다음과 같다.

**Uncertainty-based querying algorithms**

- LC(least confidence) : 모델이 예측한 라벨의 확률이 가장 불확실한 데이터를 라벨할 데이터로 선정
    
    uncertainty는 다음과 같이 정의한다.

    $
    1-P(y*|x)
    $    
    
    y*는 예측한 라벨이고, uncertainty 가 가장 높은 x를 query 할 데이터로 선정하는 것이다.
    
- Margin : 상위 두 개의 sequence 라벨의 확률 차이를 활용한다.  차이가 작을수록 높은 불확실성을 의미한다.

    $
    P(y*|x)-P(y**|x)
    $
    
- N-best seequence entropy : 라벨값을 가질 확률 상위 n개의 확률 분포의 엔트로피를 이용, N=3 를 사용하였다.
- Dynamic N-best sequence entropy(new) : 라벨값을 가질 확률값의 합이 0.9 이상이 되도록 하는 상위 N개를 선택한다. 만약 상위 4개의 확률이 0.4,0.3,0.1,0.1 이라면 확률합이 0.9이므로 N=4가 된다.
- Word Entropy : 문장을 구성하는 모든 단어의 엔트로피 합을 문장의 엔트로피값으로 활용한다. 각 단어의 엔트로피는 각 단어가 각각 7개의 라벨을 가질 확률을 이용하여 
$Entropy(word_i) = -\sum_{j=1}^7P(y_j|w_i)logP(y_j|w_i)$ 
로 정의한다.
- Entity Entropy(new) : word entropy 의 heuristic한 version으로, 모든 단어의 엔트로피 합이 아닌 B 엔티티들, 즉 엔티티의 시작 토큰의 엔트로피 합만을 활용한다.

**Diversity-based querying algorithms**

uncertainty 기반의 방법은 모델의 성능에 좌우되기 때문에 문장 간의 유사도를 고려한 diversity 기반 알고리즘도 활용하였다. 라벨링이 된 데이터와 라벨링이 되지 않은 데이터간 complete-linkage 방법으로 유사도를 결정하고, 유사도가 가장 낮은 데이터를 라벨링할 데이터로 선정한다. 유사도 결정을 위한 문장의 representation vector는 다음 세 가지 방법을 비교하여 결정하였다.

- word similarity(new) : 문장을 구성하는 단어의 TF-IDF 벡터간 코사인 유사도
- syntax similarity(new)
- semantic similarity(new)
    
    ![Untitled](https://user-images.githubusercontent.com/69342517/216894357-b9241de3-cd7b-4c8f-af74-4afdb8b5190a.png)
    
- combined similarity(new)

**Baseline algorithms**

모든 문장이 annonation cost가 동일하지 않다는 점을 반영하기 위해 **문장의 길이를 고려**하는 방법을 baseline으로 활용하였다. “문장의 길이”는 크게 **문장을 구성하는 단어 개수**와 **문장에 포함되어 있는 concept 개수**를 고려하는 방법 두 가지가 있다. 

1) length-words : 문장을 구성하는 단어 개수가 가장 많은 문장을 information이 가장 많다고 가정하여 query 하거나,

2) length-concepts : KMCI에 정의되어 있는 concept를 가장 많이 포함한 문장을 information이 가장 많다고 가정하여 query 한다.

3) passive learning 방법인 random sampling 도 활용한다. 

### 2.4. Evaluation
기존 방법과 동일하게 testset에 대한 **F-measure**를 시각화하고, 기존 방법과 다르게 문장 내 단어에 따른 annotation 비용의 차이를 고려하기 위해 단어에 따른 F-measure도 시각화하였다. F-measure와 더불어 **ALC(Area under Curve)**값도 metric으로 이용하였다. 또한, 쿼리 방법의 특징을 고려하기 위해 총 문장 개수 대비 문장 내 엔티티의 개수에 따른 **entity count curve**와 라벨링 된 문장 개수 대비 문장 내 단어의 개수에 따른 **sentence length curve** 를 시각화하였다. 5-fold CV를 사용하여 평균값을 이용하였다. 

# 3. Results