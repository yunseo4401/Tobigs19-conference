# Tobigs19-conference
## 광고, 낚시, 허위 없는 깨끗한 기사 만들기 📰

투빅스 19기 NLP 프로젝트로 다양한 뉴스에 대해서 잘못된 기사에 대해 깨끗하게 만들어 주기 위한 프로젝트를 진행했습니다.

![](https://github.com/choitaesoon/Tobigs19-conference/assets/113870266/bbe65eb5-9e0c-4163-95b1-fbe8574e7d4d)

크게 3개의 주제에 대해서 분석을 진행하였고 광고, 낚시, 허위 관련 뉴스를 올바르게 판단하는 모델을 개발했습니다.
각 파트별로 코드를 정리해두었습니다.

## 발표 🙋

컨퍼런스 발표 ppt입니다. 자세한 분석 내용은 아래 링크를 통해 확인해주세요!  
- [Slide](https://docs.google.com/viewer?url=https://github.com/choitaesoon/Tobigs19-conference/blob/main/%EA%B9%A8%EB%81%97%ED%95%9C%EA%B8%B0%EC%82%AC%EB%A7%8C%EB%93%A4%EA%B8%B0_%EC%9E%90%EC%97%B0%EC%9D%B8%EC%9D%B4%EB%8B%A4.pdf?raw=T)


## 멤버 🧑‍🤝‍🧑

- 본 프로젝트에는 [빅데이터 분석 및 인공지능 대표 연합동아리 ToBig's](http://www.datamarket.kr/xe/) 멤버들이 참여하였습니다.

|기수|이름|
|:-----:|:-----:|
|18기|[강효은]|
|18기|[김성훈]|
|18기|[현승현]|
|19기|[김윤서]|
|19기|[김은호]|
|19기|[윤세휘]|
|19기|[최태순]|

## 내용 정리

### clean_news
1. consider both title and content
2. title : learning clickbaitClass
3. content : textRank + ko-BART + Jaccard Similarity
4. model : FNN model(title_prob + content_prob)
5. making library

### fake_news
1. news data & factcheck data crawling
2. crawling data preprocessing(summerization + labeling)
3. STS + NLI model(by KLUE & roBERTa)
4. final sentence and judgment of T/F about news
5. making library

## 참고 자료
[](https://github.com/Huffon/klue-transformers-tutorial.git)https://github.com/Huffon/klue-transformers-tutorial.git

