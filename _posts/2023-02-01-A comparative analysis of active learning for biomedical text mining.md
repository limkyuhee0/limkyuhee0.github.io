---
title: "[논문 리뷰] A comparative analysis of AL for biomedical text mining"
categories:
  - AL
tags:
  - AL
  - NLP
  - Medical Domain
---


# 1. Intro

 의료정보(clinical information)은 구조화되지 않은 형태로 존재한다. 정보 활요을 위해서는 정돈된 데이터(organized data)가 필요하기 때문에 NLP 중에서도 IE(information extraction)의 중요성이 대두되고 있다. 하지만 동음이의어 구분 불가, clinical 데이터셋 자체의 문제(e.g. unstructured, ungrammatical, divided)가 있다. 

기존에는 ShARe/CLEF 2013 eHealth Evaluation Lab, i2b2/VA challenge 등 ML 기반 알고리즘과 rule 기반의 알고리즘이 존재했다. 주로 ML 기반의 알고리즘이 나은 성능을 보였다. 하지만 의료 텍스트 특성 상 라벨 부여를 위한 비용이 상당하고, 일반 도메인에 적용이 어렵다. 

따라서 AL을 포함한 semi-supervised learning이 도입되었다. AL은 적은 양의 데이터로부터 모델을 훈련시키고 query strategy를 이용하여 라벨링할 informative 데이터를 추출하여 사람에게 전달하면 사람은 이에 라벨을 부여하여 매 iteration마다 supervised ML-based model을 생성하여 훈련하는 방식이다.

해당 논문에서는 AL이 clinical data에서 효율적인지(라벨링에 소요되는 시간을 줄여줄 수 있는지), 그리고 어떤 AL 기법이 라벨링 소요 시간을 가장 많이 줄여주는지를 알아보고자 하였다. 

# 2. Related Work

의료 데이터를 디지털화한 electronic medical records(EMR)의 등장에 따라 biomedical literature(e.g. books, scientific articles)과 free-text clinical narratives(e.g. synopses) 등의 의료 데이터로부터 주요 데이터를 추출하는 IE 가 많이 연구되고 있다. 주요 목적은 주요 아이디어 추출의 자동화와 함께 추출된 주요 아이디어의 품질을 보장하는 것이다. 

### 2.1. IE from Biomedical Corpus

비구조화된 raw clinical data로부터 바로  NLP 기반의 방법을 활용할 수 없기 때문에 유용한 정보를 포함한 단어를 추출해 내는 과정이 필요하다. 이는 다음과 같은 두 가지 방법으로 수행 가능하다.

**Dictionary-based Methods**

패턴을 이용하여 사전에 정의된 리스트에 포함된 단어를 추출해 내는 방법이다. 사전에 정의된 리스트는 SNOMED CT와 UMLS 등의 대량 딕셔너리(dictionary)가 공개되어 있다. 이는 엔티티를 normalize 하고 syntactic, semantic 수준의 엔티티와 연결하여 활용할 수 있다는 장점이 있지만 해당 도메인에서만 활용할 수 있다는 coverage 문제가 있다.

**Rule-based Methods**

Rule based methods는 manually created rules 을 이용하여 natural language에서의 패턴을 추출하고자 한다. 최근에 발견된 요소를 포함하지 않는 등의 단점을 갖는 dictionary based 기법의 단점을 문장 구조 기반의 데이터 추출을 통해 극복하고자 한다. 대표적인 방법으로는 기존의 term, rule, 그리고 shallow parsing methods을 이용하여 biomedical information을 추출하고자 하는 방법과 BioTeks 등의 biomedical corpus로부터 biomedical information 추출 기법 등이 있다. 하지만 도메인 전문가가 필요하고 다른 도메인에 적용할 수 없다는 단점이 있다. 

### 2.2. ML

2.1.의 단점들을 보완하기 위해 ML을 활용한다. 가장 많이 활용되는 방법은 SVM과 CRF 이다. 하지만 이러한 모델의 훈련을 위해서는 많은 양의 labeled 데이터가 필요하다. 이에 들어가는 노력을 줄이기 위해 AL 을 이용하여 less-annotated training data를 이용하여 더 나은 성능을 보인다. AL과 더불어 Semi-supervised learning은 라벨 부여가 안 된 데이터를 iterative interaction 과정에서 라벨을 부여하는 self-training 과정을 거친다. 모델에 word embedding 등 단어들의 representation 벡터를 모델에 feed 하기 위해 representation learning이라는 개념도 소개되었다. 

### 2.3. NLP

- Biomedical NER(BioNER)
    
    ![image](https://user-images.githubusercontent.com/69342517/215981748-c37011b9-40c6-421c-8e04-35b60450b9f7.png)
    

Biomedical 텍스트에서는 정보 중복이 발생하고, NN 모델은 feature를 추출하고자 하기 때문에 주요 정보가 수집되지 못해서 information loss 가 발생할 수 있다. 따라서 데이터에서 주요한 부분을 모델이 알게 하는 것이 중요하다.

- MT(Machine Translation)
    - attention focusing mechanism - decoder 모델을 이용하여 초기에 주어진 텍스트의 주요 부분을 이용하여 decoding 과정에서 focus 를 생성하고 information loss 를 줄일 수 있다.
    - attention based BiLSTM-CRF :attention-focused mechanism 을 이용하여 tagging inconsistency 문제 최적화 - CHEMDNER & CDR corpora에서 가장 좋은 성능을 보인다.
- Fine Tuning
    - training ELMo on biomedical corpora for BioNER task
    - training BERT on scientific text
    - BioBERT : training BERT on PMC abstracts and articles in PubMed
    - BLUE(Biomedical Language Understanding Evaluation) : resources for evaluating and analyzing natural biomedical language representation models

### 2.4. AL

사람 개입(라벨링 과정)을 최소화하여 필요한 개수만을 이용한 모델의 학습을 통해 모델의 효율성 최대화

### 2.5. AL in Clinical Domain

Random Sampling 이 가장 흔하게 이용된다.

- de-identifying clinical records
- clinical text classification
- clinical NER

⇒ 해당 논문에서는 AL을 위해 annotate하는 cost 를 알아보고자 한다.

# 3. Methodology

### 3.1. Dataset

- Dataset
    <figure class="half">
        <img src="https://user-images.githubusercontent.com/69342517/215981601-23890660-5a39-424d-8467-d34cb28019c6.png" width="450"/>
        <img src="https://user-images.githubusercontent.com/69342517/215981671-1470aaad-a648-46de-872d-0f5386530b16.png" width="200"/>
    <figure>


### 3.2. & 3.3. AL Query Strategies

- RS(Random Sampling) : 전체 데이터셋 중 학습할 데이터를 랜덤으로 선정; 모든 데이터는 동일한 확률로 뽑힌다.
- Uncertainty Sampling
    - LC(least confidence) : 라벨이 가장 불확실한 데이터를 라벨링해야 할 데이터(most informative data)로 선정
    - Margin : 가장 높은 가능성을 가진 라벨을 가질 확률과 그 다음 높은 가능성을 가진 라벨을 가질 확률 간의 차이가 큰 데이터를 라벨링 할 데이터(most informative data)로 선정
- IDD(Informative Diversity and Density) : calculate information density of an instance x - 데이터의 구조를 고려한다.
- MRD(Maximum Representativeness-Diversity) : 미리 라벨링할 샘플들을 batch 형태로 추출해 놓아서 모델의 학습을 기다리지 않아도 된다는 장점을 가진다.
    
    > The most representative is to mark various samples in the current batch and then add them to the training set. This method could prevent experts from waiting nutil the learning model is on the current set of tags, and then the next batch of samples selects the tags using one of the above query strategies.
    > 

### 3.4. Feature Extraction Methods

TF-IDF(빈도만 고려) vs FastText(임베딩 라이브러리) vs BERT, ELMo

### 3.5. ML Methods

SVM, KNN, NB vs Gradient boosts based on RT(XGBoost, CatBoost), Ensemble Methods(RF, AdaBoost)

# 4. Results & Discussion

![image](https://user-images.githubusercontent.com/69342517/215981523-0909110a-f127-44f9-94db-52f6e6a6ad93.png)


전반적으로 ML 기반의 모델(특히 CatBoost는 모든 케이스에서 일관된 성능을 보였다. DT 기반의 gradient boost 이기 때문에 나중 tree에서 오류 수정 가능성이 높고, 다른 gradient boost 기법에 비해 hyper-parameter 변경 시에 더 효율적이다.)이 좋은 성능을 보였고 AL 이 passive learning 알고리즘보다 조금 더 좋은 성능을 보였다. query strategy 중에서는 Margin과 LC가 전반적으로 좋은 성능을 보였다. SVM은 데이터간의 상관관계에 영향을 크게 받아 다른 알고리즘에 비해 성능이 좋지 않았다. 

Uncertainty sampling 에 해당하는 margin 과 LC 는 classfication 에러를 줄이고자 하는 목적을 가지고, 이는 해당 논문의 목적과 부합하기 때문에 좋은 성능을 보인다. IDD나 MRD는 효율적으로 사전 작업할 수 있다면(efficiently pre-computed; ~~먼소리지..~~) 더 좋은 성능을 보일 것이라 판단된다.

AL이 passive learning 에 비해 더 좋은 성능을 보이는 이유는 unbalance of the dataset 때문이다. 데이터의 상대적인 양이 아닌 주요한 데이터를 이용한 학습이 모델의 성능을 결정짓는다. 

# 5. Conclusions

전반적으로 AL 이 passive learning 보다 좋은 성능을 보였지만, 문장의 길이를 고려하였을 때 그 효율성이 감소하였다. 따라서 NER 보다는 dataset classification 에서 가장 좋은 효율을 보인다. 

CatBoost 가 Uncertainty sampling 기반의 AL과 함께 이용될 때 가장 좋은 성능을 보이고 AL이 ML의 일부이기 때문에 도메인 지식에 대한 필요성이 크지 않다.