---
title: "[논문 리뷰] Semi-supervised active learning for modeling medical concepts from free text"
use_math: true
categories:
  - AL
tags:
  - AL
  - NLP
  - Medical Domain
---


***
- Date : 2007


# Abstract

medical concepts 를 추출하고자 하는 목적을 가지고 QBC 를 응용한 AL을 사용한다. QBC와 다르게 해당 논문에서 제안하는 모델은 라벨할 데이터를 query 할 때 unlabeled 데이터까지 반영해서 train set과 관련된 distribution 또한 학습하도록 한다. 특히 test set의 distribution이 train set과 다를 때 생기는 문제점을 보완한다. 

# 1. Introduction

concept learning 과 information retrieval 시스템에서 유저의 개입은 효율성을 증진시킨다. 특히 active learning 시스템에서는 라벨링 되지 않은 대량의 데이터를 바탕으로 유저가 라벨링할 데이터를 선정한다.

라벨이 주어진 데이터가 적고 라벨이 주어지지 않은 데이터를 얻기 쉬운 경우에 AL을 사용하는데 특히 text-based language process 에서 가장 많이 발생한다. 해당 논문에서는 의료 text document, 특히 EMR 을 사용한다.

**AL Theory**

AL 을 이용하면 필요한 데이터의 양이 $O(1/e)$ 에서 $O(log(1/e))$로 감소한다. e는 보장할 수 있는 error 의 정도이다.

**Related Approaches**

기존에 사용한 AL 방법으로는 

US(uncertainty sampling) : 모델이 라벨을 부여하기 가장 불확실한 데이터를 query 한다. 하지만 noisy,rare 데이터를 선정할 가능성이 높다는 단점이 있다. 

그 중에서도 예상 에러를 감소시키는 데이터를 선정하는 방법이 있지만 예상 에러를 얻기 힘들고 이 과정에서도 샘플링이 필요하다는 문제가 있다. 또한 매번 데이터를 query할 때마다 모델을 다시 학습해야 한다는 문제가 있다. 

QBC는 size of version space(parameter 개수에 의해 결정)를 줄여주는 데이터를 라벨링할 데이터로 선정한다. 이를 위해 여러 개의 독립적으로 학습된 모델들로부터 예측된 라벨이 가장 일치하지 않는 데이터를 라벨링할 데이터로 선정한다. 해당 모델은 데이터가 업데이트될 때마다 모델 재학습이 필요하지 않다는 장점이 있다.

해당 연구에서 제안하는 모델은 query 데이터를 선정하는 것 외에 unlabelled 데이터를 이용할 수 있는 방안을 제시한다. 해당 모델은 Mutual Information을 최대화 하는 기준으로 AL을 사용하고, 데이터 분포를 고려한 데이터 포인트를 선정한다. 이는 데이터로부터 원하는 지점의 데이터를 선정할 수 있도록 한다.

# 2. Unlabeled Data & AL

기존 AL이 query를 위해 각 데이터에 부여하는 score은 서로 다른 unlabeled 데이터와 독립적이었다. 이 점을 보완하기 위해 regression 문제에 transductive learning 을 AL에 활용한 연구가 있다. unlabeled 데이터의 정보를 활용하기 위해 query pool 뿐만 아니라 test set과 같은 unlabeled dataset의 어느 subset이든 활용한다. 

어떤 모델이든 데이터의 특정 부분에 집중하고자 때문에 이러한 부분을 찾는 과정으로 볼 수 있다. 이러한 과정을 통해 1) train test 데이터 분포간 차이 문제나 2) concept drift 문제, 3) 고정되지 않은 데이터 분포 등의 문제를 해결할 수 있다. 

# 3. Formulation

Y : incidence of particular concept of interest(라벨 말하는듯)

X : text representation

$X_i$, i < D; D : index set

p : 텍스트 분포

p(Y=y,X=x, $\theta=\theta$) - $y$ : concept incidence, $x$ : input, $\theta$ : 모델(파라미터)

라벨링할 데이터 $x_i$의 결정은 그 데이터 포인터 뿐만 아니라 다른 unlabeled data에까지 의존한다. $X_i, Y_i$ 쌍으로 존재하고 라벨링 되지 않은 데이터에서 $Y_i$ 는 없음
$argmax_{i\in U}H(Y_i|x_D, y_L)-H(Y_i|\theta$,$x_D$,$y_L)$
위와 같은 최적화 문제로 정의할 수 있다. H는 엔트로피 값이고, D는 전체 데이터셋, L과 U는 각각 라벨링 된 데이터셋과 라벨링 되지 않은 데이터셋이다. $\theta$에 대한 계산을 위해 $\theta$를 K 개로(K개의 모델을 comittee로 활용한다는 뜻) 제한하면 다음과 같은 식과 동일해진다.
$argmax−\sum_{k=1}^K\sum_{y_i}p(y_i ,\theta_k |x_D , y_L )log\sum_{j=1}^K p(y_i ,θ_j |x_D , y_L ) + \sum_{k=1}^K\sum_{y_i} p(y_i ,\theta_k |x_D , y_L )log p(y_i |θ_k,x_D , y_L )$
이는 conditional KL divergence에 의해

![image](https://user-images.githubusercontent.com/69342517/216835809-1d17fb97-4cc4-4389-bf7c-0bc50b5e8297.png)

와 같이 바뀐다. KL divergence는 두 확률분포의 차이를 계산하여 데이터를 표현한 확률 모델이 정보 손실이 얼마나 있었는지를 측정하는 메트릭이다. 이러한 KL divergence 가 큰 값을 query 하는 것이 목표이다. 

요약하자면 다음과 같은 과정을 거친다.

> 1. 데이터 포인트를 랜덤으로 선정
>2. 라벨이 있는 데이터를 랜덤으로 K개의 모델에서 학습시켜 MAP(Maximum-a-Posteriori) 예측값을 계산한다.
>3. 라벨이 없는 데이터 각각에 대해 KL divergence 값을 구하고 가장 큰 데이터를 query 할 데이터로 선정하여 라벨링하고, train set으로 추가시킨다.
>4. K개의 모델에서 MAP 예측값을 업데이트한다.
 

> MAP([https://decisionboundary.tistory.com/5](https://decisionboundary.tistory.com/5))
>obs = 머리카락 길이
>- **MLE(Maximum Likelihood Estimation):** MLE는 남자에게서 해당 길이의 머리카락이 나올 확률 $P(obs | M)$과 여자에게서 해당 머리카락이 나올 확률 $P(obs | W)$을 비교해서 가장 확률이 큰, 즉 가능도가 가장 큰 성별을 선택합니다.
>- **MAP(Maximum A Posteriori):** MAP은 obs라는 머리카락이 발견되었는데 그 머리카락이 남자의 것일 확률 $P(M | obs)$, 그것이 여자 것일 확률 $P(W | obs)$를 비교해서 둘 중 큰 값을 갖는 성별을 선택하는 방법입니다. 즉, 사후확률(posterior prabability)를 최대화시키는 방법으로서 MAP에서 사후확률을 계산할 때 베이즈 정리가 이용됩니다!


기존 QBC에서의 KL Divergence 는 다음과 같다.

![image](https://user-images.githubusercontent.com/69342517/216835769-cdf228c2-6658-4a0b-92eb-906497187b78.png)

라벨 변수만 고려하여 KL divergence의 합을 계산하는 QBC와 다르게 해당 연구에서 제안하는 모델은 라벨과 모델 변수 모두를 고려한다. 

# 4. Semi-Supervised AL

**가정**

해당 연구에서 가정하는 분포는 $p(x_i|y_i$,$\theta)$ 이다. 이 분포는 임의적이기 때문에 다른 변수를 포함할 수 있다. 즉, 각각의 데이터쌍 (X,Y)는 독립적이다. 따라서 
$p(y_i |θ_j , x_D , y_L )=p(y_i |θ_j , x_i )$ 
와 
$p(θ_k |x_D , y_L )=p(θ_j |x_U )$ 
를 가정한다. 모델의 형태는 다음과 같다.

![image](https://user-images.githubusercontent.com/69342517/216835830-529892e4-ff36-4a49-92e1-d65560c50613.png)

### 4.1. Aiming Model Descriptive Power

$x_U$ 중 query point는 가장 informative 한 데이터이기도 하지만 data distribution을 대표하는 데이터이기도 하다. 따라서 $x_U$는 데이터 분포에 대한 예측을 제공한다. test 데이터셋에 대한 정보가 있는 경우에 AL을 이용하여 test 데이터셋 분포를 파악하고, 이를 통해서 test 데이터셋과 train 데이터셋의 차이가 있을 때 발생하는 문제를 해결함으로써 test 데이터셋에 대해서도 좋은 결과를 낼 수 있다. 또한, 모델을 정제하는데에 활용할 수 있다. 대표적으로, 일반적인 데이터에 대해 학습한 모델을 특정 데이터에 모델링하는데 AL 이 도움을 줄 수 있다. 

# 5. Experimental Evaluation

EMR 데이터셋을 이용하였고, 환자나 문서를 특정 증상에 따라 분류하는 것을 목표로 한다.

### 5.1. Representation and Datasets

문장 단위로 작업하였고 따라서 $X_i$ : sentence representation

6개의 medical concepts of heart disease 키워드가 포함된 문장을 활용하였고, 일부가 manual하게 라벨링되었다. 

![image](https://user-images.githubusercontent.com/69342517/216835843-37b99f10-7486-473a-99d5-b9576c67e035.png)

라벨은 T,F 로, T의 경우 concept가 문장 내에 존재하고 유의할 때, F의 경우 문장에 concept 가 존재하지 않거나 문장 내에 존재해도 유의하지 않을때 부여된다. 이는 F(문장 내에 존재해도 유의하지 않을때)와 N/A(문장에 concept 가 존재하지 않을때)로 나누어 활용할 수 있다.

### 5.2. Experimental Settings

15개의 모델을 활용하였고 데이터셋은 

test(20%), train(80%; 8%는 랜덤하게 선정되어 초기 학습에 활용, 92%는 AL에 활용, active learning 문장 중에서 75%만 활용하였다.)

10 epochs, K개의 결과값의 평균을 사용하되, 가장 좋은 모델을 선정하기 위해 cross-validation set을 이용하였다. 

문장 input 당 단어 개수는 20으로 고정, 20개의 단어는 모델 훈련 중에 가장 mutual information을 최대화시키는 단어들로 선정

### 5.3. Results

테스트 결과는 QBC와 random sentence selection과 비교하였고, KL Divergence 계산에서의 $x_U$또한 active set에서의 데이터일때와 test set에서의 데이터일 때의 결과를 비교하였다.(query는 active set에서만)

performance는 초기 모델 대비 잘못 예측된 개수의 비율(RE) 값을 활용한다.

![image](https://user-images.githubusercontent.com/69342517/216835874-1c7844f5-af58-4d36-b39b-b87742be169b.png)

QBC보다 성능이 좋음을 확인했다. 해당 방법은 active set에서가 test set에서보다 더 좋은 결과를 보였다.

아래 그림은 test set과 active set에 대한 해당 연구의 모델과 QBC의 예측 에러를 나타낸 것이다. 10 epoch 평균을 나타내며, epoch이 지나면서 test set에 대한 에러가 증가하는 것은 overfitting 을 의미한다. 해당 연구의 모델이 항상 QBC보다 에러가 낮음을 볼 수 있다. 

![image](https://user-images.githubusercontent.com/69342517/216835883-f8453e60-42f4-4a62-af2a-97d4979e7bab.png)

# Conclusion

unlabelled 데이터의 활용을 증가시킴으로써 QBC에 비해 더 좋은 성능을 보이고 더 필요한 input 데이터를 선별할 수 있음을 확인하였다. 의료 데이터에서는 특정 도메인에 대한 corpus가 있어도 다른 도메인에 적용하기 힘든 경우가 있다. 따라서 generic model을 훈련시키 후 새로운 도메인에 대한 데이터를 이용하여 fine-tuning하고 이 과정에 AL을 활용하는 것이 데이터 분포가 일정하지 않은 데이터나 의료 데이터와 같이 train, test 데이터의 분포가 다를 때 좋은 결과를 도출할 수 있을 것이다.