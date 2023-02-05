---
title: "[논문 리뷰] MMR-based active machine learning for bio named entity recognition(2006)"
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

해당 연구에서는 모델의 **uncertainty** 뿐만아니라 데이터의 **diversity**도 고려하여 두 방법을 모두 혼합한 MMR(Maximal Marginal Relevence) 방법을 활용하여 샘플링한다. 이를 이용하여 **의료 도메인의 NER 태스크**를 수행하고자한다. 

# Introduction

**Biomedical Text mining 중에서 NER 태스크**를 해결하기 위한 여러 머신러닝 방법들은 학습을 위한 training set이 필요하다. Training set 생성을 위한 annotation 단계에는 많은 시간과 비용이 소모된다.

이를 보완하기 위해 Active Learning(AL) 이 사용된다. 기존 AL은 크게 **uncertainty** 기반과 **committee** 기반으로 나뉜다. Uncertainty 기반의 방법은 classifier 하나가 라벨이 부여되지 않은 샘플들 각각에 대한 uncertainty 점수를 부여하는 방식이다. Committee 기반의 방법은 여러 classifier로 구성된 committee가 각 샘플에 대한 라벨을 예측하고, 라벨이 다르게 예측된 비중이 가장 큰 샘플이 선택된다.

해당 연구에서는 entropy 기반의 방법을 이용하여 uncertainty를 정량화하고, 선택된 샘플이 학습해야 하는 feature의 다양한 요소를 학습할 수 있도록 해주는 데이터를 선택하기 위해 문장간의 divergence 측정을 통해 가장 유사도가 낮은 문장을 선택하고자 한다. divergence 측정은 문장이 얼마나 새로운 feature를 포함하고 대표적인지(representative)를 나타내고, 이러한 지표를 이용함으로써 샘플의 diversity를 고려할 수 있다. 

해당 연구에서 MMR은 기존 CRF 모델 기반의 NER 시스템인 POSBIOTM/NER 에 도입하였다. 

# MMR-based AL for Biomedical NER

### AL

해당 연구에서는 다음과 같은 과정으로 AL을 기존의 POSBIOTM/NER 시스템에 적용하였다. 1) 모듈 M을 이용하여 pool U의 샘플들의 라벨을 예측한다. 2) AL 기법 S에 의해 부여된 점수가 threshold th 이상이라면  점검 후 train set에 추가한다. 3) train set에 일정량의 데이터가 쌓이면 M을 재학습한다. 

### Uncertainty-based Sample Selection

해당 연구에서는 인풋으로 특정 **sequence(o)가 주어졌을 때** **state sequence(S)** 를 예측하고자 한다. State sequence란 각 토큰의 라벨을 sequence 단위로 묶어서 연속적인 단위로 예측하는 것이라고 생각하면 된다. 

따라서 구하고자 하는 식은 $p(s|o), s\in S$ 이고, uncertainty 계산을 위한 엔트로피는 

$
H = -\sum_s P(s|o)log_2[P(s|o)]
$

하지만 input이 sequence라는 점을 생각하면 sequence의 길이가 늘어남에 따라 가능한 s 의 경우의 수가 기하급수적으로 증가하게 된다. 따라서 해당 연구에서는 **N-best Viterbi search** 를 이용해서 가장 확률값이 높은 상위 N개의 s를 이용한다. 이때 엔트로피 $H(N)$ 는 다음과 같다. 

$
H(N) = \sum_{i=1}^N \frac{P(s|o)}{\sum_{i=1}^NP(s|o)} log_2[\frac{P(s|o)}{\sum_{i=1}^NP(s|o)}]
$

이 값은 [0, $-log_2\frac{1}{N}$ ] 이기 때문에 [0,1]로 변환해주기 위해 다음과 같은 식을 사용해도 된다.

$
H(N)^` = \frac{H(N)}{-log\frac{1}{N}}
$

### Diversity-based Sample Selection

문장 구조의 유사성을 반영하여 최대한 다양한 샘플을 추출할 수 있도록 한다. 먼저 각 문장을 **(NP chunk, POS tag, word)** 형태의 3차원 벡터로 나타낸다. 가령 “boy”라는 단어는 [NP,NN,boy] 벡터로 표현된다. 이러한 단어 벡터간 유사성을 다음과 같이 정의한다. 

$
sim(
\overrightarrow{w_1}\circ
\overrightarrow{w_2}) = \frac{2*Depth(\overrightarrow{w_1},
\overrightarrow{w_2})}{Depth(\overrightarrow{w_1})+Depth(
\overrightarrow{w_2})}
$

이때 Depth는 비교하고자 하는 두 단어의 벡터 중 몇 차원만큼 유사한지를 나타낸다. 가령 “boy” 와 비교하고자 하는 단어의 벡터 표현이 boy의 벡터 표현과 NP만 동일하다면 Depth는 1 이 되는 것이다. 

문장의 경우 위와 같은 3차원 단어벡터들로 구성된 벡터간 코사인 유사도로 두 문장이 얼마나 유사한지 정의한다.

$
similarity(\overrightarrow{S_1},
\overrightarrow{S_2}) = \frac{\overrightarrow{S_1}\circ
\overrightarrow{S_2}}{\sqrt{\overrightarrow{S_1}\circ
\overrightarrow{S_1}}\sqrt{\overrightarrow{S_2}\circ
\overrightarrow{S_2}}}, \overrightarrow{S_1}\circ
\overrightarrow{S_2} = \sum_i\sum_jsim(\overrightarrow{w_{1i}}\circ
\overrightarrow{w_{2j}})
$

### MMR Combination of Sample Selection

$score(s_i) = \lambda * Uncertainty(s_i,M) - (1-\lambda) * max_{s_j\in T_M} Similarity(s_i,s_j)$

$s_i$는 pool U에서 선택될 수 있는 문장들이고 Uncertainty는 모델의 엔트로피값, Similarity는 Training set의 문장 $s_j$와 $s_i$ 사이의 divergence 를 의미한다. 위의 점수가 높을수록 **uncertain**하고 **training set의 문장들과 구성이 다른 문장**이다. 

# Experiment & Discussion

### Experiment Setup

해당 연구에서는 pool 에서 라벨링 할 데이터를 선택하는 pool-based sample selection을 수행한다. 

- 데이터 : JNLPBA(train : 2000 Medline Abstract, test : 404 GENIA Abstract)
    - 기존 NER 모델 학습을 위해 100개 데이터 사용, 나머지는 pool로 활용하였다.
    - 매 iteration 마다 k(1000~17000, step size = 1000) 개의 데이터를 선택하였다.
- AL 기법 : Random / Uncertainty / Diversity / Normalized Entropy + Diversity
- 고정 하이퍼 파라미터 : $\lambda$ = 0.8 , N = 3

### Results & Analysis

지표는 F-score을 활용하였다. 

![Untitled](https://user-images.githubusercontent.com/69342517/216845530-fcc8a89d-938a-4248-a8d1-f9d464f08a7f.png){.align-center}

- 초기 NER 모델 : 52.64
- 학습된 초기 NER 모델 : 67.19
- 가장 좋은 성능 : 67.31(17000 samples)
- 다른 4개의 AL 모델(uncertainty / uncertainty + diversity / normalized uncertainty + diversity / random)이 더 나은 성능을 보이는 것을 확인하였다.
- 모델이 모두 같은 pool로부터 학습을 하기 때문에 pool의 샘플을 모두 활용할수록 비슷한 성능을 보이게 된다.
- 4000개 샘플을 선택했을 때 uncertainty 와 combined 의 성능이 비슷하지만 11000개의 샘플을 선택했을 때는 combined 의 성능이 더 좋은 것을 보아, 선택하는 샘플의 개수가 많을수록 diversity를 고려하는 것이 모델의 성능을 높인다는 것을 알 수 있다.
- Normalized Uncertainty를 활용하는 경우 12000개 샘플을 선택할 때까지 가장 좋은 성능을 보였다. 13000개의 샘플을 선택할 때는 random 과 비슷했고, 14000개의 샘플을 선택했을 때는 가장 성능이 안 좋았다.
- 모든 데이터를 활용하였을 때 최종 점수인 67.19를 달성하기 위해 combined 방법은 13000개의 데이터를 활용하였다. 즉 필요한 데이터의 양이  24.64% 감소하였다. normalized combined 방법은 11000개의 데이터를 활용하여 필요한 데이터의 양이 35.43% 감소하였다.