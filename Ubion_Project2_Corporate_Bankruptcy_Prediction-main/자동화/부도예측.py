import pandas as pd
import numpy as np
import time
import re
import requests
import pickle
import joblib
import lxml
import pymysql
import warnings

from eunjeon import Mecab
from tensorflow.keras.preprocessing import sequence
from keras.models import load_model
from selenium import webdriver
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from datetime import datetime

warnings.filterwarnings('ignore')
## 사용할 데이터는 ts2000에서 가져오고 순서가 자본(연결) 자본(개별) 이렇게 연결이 먼저 오는 형태로 있어야 합니다.
search = True
while search == True:
    _db = pymysql.connect(
    user = "newuser", # mysql 아이디
    passwd = "dbswo123", # mysql 비밀번호
    host ="172.30.1.12", # 내 컴퓨터
    db = "ubion"
    )

    기업이름 = input("검색하고 싶은 기업 이름을 넣어주세요. (Stop을 입력하면 프로그램이 종료됩니다.) : ")

    cursor = _db.cursor(pymysql.cursors.DictCursor)

    sql = "SELECT * FROM 사업보고서_20년도 \
        WHERE 회사명='{}'".format(기업이름)
    cursor.execute(sql, )
    result = cursor.fetchall()

    cursor.execute(sql)
    result = cursor.fetchall()
    if len(result) == 1:
        사용할재무비율 = pd.DataFrame(result)
        search = False
    elif 기업이름 == 'Stop':
        quit()
    else:
        print("데이터베이스에 존재하지 않는 기업입니다.")

## 가공한 데이터프레임에서 회사명, 회계년도, 최종적으로 사용할 재무비율만 가져오고 회사명에 (주)는 필요없기에 제거합니다.
## 상장폐지는 상폐일 3개월전쯤에 확정. 현재 알아보고 싶은 기업이 상장폐지 직전일 수도 있기 때문에 뉴스 수집기간은 (현재시점-3개월)을 기준으로 과거 1년의 데이터를 수집합니다.
dif_3m = relativedelta(months=3)
dif_12m = relativedelta(months=12)

오늘날짜 = datetime.now().date()

## 네이버 크롤링시에 url에 기간을 넣어줘야하므로 사용할 변수들을 생성합니다.
기사수집종료날짜 = 오늘날짜 - dif_3m
기사수집시작날짜 = 기사수집종료날짜 - dif_12m

종료연도 = str(기사수집종료날짜.year)
종료월 = str(기사수집종료날짜.month)
종료일 = str(기사수집종료날짜.day)

시작연도 = str(기사수집시작날짜.year)
시작월 = str(기사수집시작날짜.month)
시작일 = str(기사수집시작날짜.day)

## 기업별로 검색해서 뉴스를 가져온 다음 뉴스_df에 합치고 훈련한 텍스트 분류 모델을 통해 부도관련기사인지 아닌지 분류합니다.
## 이후 groupby를 통해 (기업별부도기사수 / 기업별전체기사수)를 통해 부도기사비율을 산출해서 변수로 활용하기 위한 크롤링.
column_ = ['기업', '기사발행일', '기사제목', '뉴스기사본문']
뉴스_df = pd.DataFrame(columns=column_)

driver = webdriver.Chrome()

start = (시작연도 + '.0' + 시작월 + '.0' + 시작일)
end = (종료연도 + '.0' + 종료월 + '.0' + 종료일)
start_= (시작연도 + '0' + 시작월 + '0' + 시작일)
end_ = (종료연도 + '0' + 종료월 + '0' + 종료일)

for 기업 in  사용할재무비율['회사명']:

    # 나중에 뉴스_df와 concat할 임시 df 생성 및 임시 리스트들 생성합니다.
    column_ = ['기업', '기사발행일', '기사제목', '뉴스기사본문']
    임시_df = pd.DataFrame(columns=column_)    

    # 임시_df에 들어갈 리스트 생성합니다.
    본문리스트 = []
    날짜리스트 = []
    제목리스트 = []
    기업이름 = []

    # while 종료 조건으로 쓸 리스트 생성합니다.
    newslist = []
    datelist = []
    
    page = 1

    # 페이지수가 나와있지않으므로 맨끝에 page에 10씩더해서 계속 다음페이지로 이동합니다.
    while page <200:

        url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query='+'"'+기업+'"'+'&sort=0&photo=0&field=0&pd=3&ds='+start+'&de='+end+'&cluster_rank=19&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from'+start_+'to'+end_+',a:all&start='+str(page)
        driver.get(url)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")

        # a태그중에서 class가 info인 것과 span태그에 class가 info 인 것 가져옵니다.
        news_titles = soup.select("a.info")
        dates = soup.select('span.info')

        # 네이버기사와 신문사기사 둘다 하이퍼링크가 있을경우는 기사당 news가 2개씩 네이버 기사가 없을 경우는 1개씩만 링크가 추출됩니다.
        # 어림잡아 news_titles수가 10개 미만일 경우는 기사의 수가 5개에서 10개 미만이라는 뜻으로 해석하여 10개 이상만 링크와 기사날짜를 추출
        if len(news_titles) >= 10:

            for news in news_titles:
                title = news.attrs['href']
                newslist.append(title)

            for date in dates:
                news_date = date.text
                datelist.append(news_date)

            # 네이버 검색시 page number가 다음페이지로 갈때마다 1, 11, 21 이렇게 10씩 더해지는데 다음페이지가 없을 경우 
            # 마지막 기사만을 포함한 같은 페이지를 계속 반환함. 따라서 기사와 날짜 둘 다 중복되서 저장될경우 종료합니다.
            if (newslist[-1]==newslist[-2]) & (datelist[-1]==datelist[-2]):
                break

            page += 10

            time.sleep(1)
        
        # 기사수 5개 미만시 break, 기사없는기업 리스트에 저장함
        else:
            
            break
    
    # sid=101은 네이버 경제기사를 의미하는 듯함. 경제기사만 추출
    newslist = [news for news in newslist if 'sid=101' in news]
    
    # news_titles 뉴스기사 url 리스트가 존재시
    if news_titles:

        # 뉴스기사 url 자체에서는 text가 안가져와지는 특이사항 발생, 찾아보니 네이버기사는 인터넷에 떠야 페이지가 작동하는 방식이라하여
        # 셀레니움을 통해서 뉴스기사 url 주소로 창을 띄웁니다.
        for news in newslist:
            url = news
            driver.get(url)
            
            # 뉴스기사 url에서 본문과 제목, 기사작성날짜를 리스트에 저장합니다
            try:

                날짜 = driver.find_element('xpath', '//*[@id="ct"]/div[1]/div[3]/div[1]/div/span').text
                날짜리스트.append(날짜)

                제목 = driver.find_element('xpath', '//*[@id="ct"]/div[1]/div[2]/h2').text
                제목리스트.append(제목)

                본문 = driver.find_element('xpath', '//*[@id="dic_area"]').text
                본문리스트.append(본문)
                time.sleep(1)

                기업이름.append(기업)
            
            ## 로딩이 안되서 데이터를 못가져올 경우를 대비해 sleep 3초 주고 다시 시도
            except:

                time.sleep(3)

                try:

                    날짜 = driver.find_element('xpath', '//*[@id="ct"]/div[1]/div[3]/div[1]/div/span').text
                    날짜리스트.append(날짜)

                    제목 = driver.find_element('xpath', '//*[@id="ct"]/div[1]/div[2]/h2').text
                    제목리스트.append(제목)

                    본문 = driver.find_element('xpath', '//*[@id="dic_area"]').text
                    본문리스트.append(본문)
                    time.sleep(1)

                    기업이름.append(기업)
                
                # 그래도 데이터를 가져오지 못하는 경우는 페이지에 문제가 있다고 판단하여 PASS
                except:

                    pass

    # 혹시나 뉴스기사 url 리스트가 없을 경우는 pass                
    else:
        pass

    # 기업별로 가져온 날짜와 본문, 제목, 기업이름을 임시 데이터프레임으로 저장    
    임시_df['기사발행일'] = 날짜리스트
    임시_df['뉴스기사본문'] = 본문리스트
    임시_df['기사제목'] = 제목리스트
    임시_df['기업'] = 기업이름
    
    # 임시 데이터프레임을 뉴스 데이터프레임에 아래로 결합
    뉴스_df = pd.concat([뉴스_df, 임시_df])

# 앞에서 while 종료 조건이 같은 기사 2번저장인데 이럴 경우 중복으로 저장이 되야 종료되기때문에
# 중복기사 행 제거
뉴스_df.drop_duplicates(inplace=True)
driver.close()

## 네이버 경제기사에 해당하지만 내용이 적거나 광고 내용으로 기사 분류에 혼동을 줄 수 있는 기사 제목들의 일부를 가져온 것 입니다.
제거할기사제목 = ['증시 일정', '증시일정', '장마감후', '장 마감 후', '장마감공시', '증시 캘린더', '재송', '투자정보', \
    '코스닥 기업공시', '장중 주요', '코스닥 3분기 결산', '주요 뉴스 및 공시', '기업공시',' 기업 공시', '장 종료 후', \
        '주요 정보', '오전 공시', '투자정보', '기업 공시', '오늘의 메모', '재테크 캘린더', '추천주 정리', '희망복원 주식클리닉', \
        '\d{1,2}월 \d{1,2}일', '오늘의 주요 공시', '코스닥 공시', '코스닥 메모', '대박 공모주', '<표>', '오늘의 리포트', \
        '주식상담소', '주식왕 따라잡기', '주식컨설팅', '\d{4} 증시', '춤추는 테마주', '개장시황', \
        '폭등신호 터진', '대폭등', '매드머니', '주담과 Q&A', '굿바이 \d{4}', '지금 당장 사라', '폭등주', \
        '\d{4}%', '김정일 사망', '종목대탐험', '종목신호등', 'VIP CLUB', '수급유망주', '기상도', '중소형주', \
        '국민주식고충처리반', '유망주', '머니Q', '시초가잡아라', '기관 Q&A', '부동산에 투자하려면', '베스트애널리스트', \
        '코스닥협회장 취임', '코스닥협회', '부동산에 투자하는 방법', '국가품질 경영대회', '증시일정', '머니Q', '추천종목', '티타임 공략주', \
        '수급유망주', '종목배틀', '\d{4} 증시 결산', '기업설명회', '종목신호등', 'VIP CLUB 추천주', '내일장 공략주 10선', '주간컨센서스동향', \
        '주담과 Q&A', '조회공시', '주가급등 사유', '관련株들', '게임株', '이시각 Up&Down', '장내 매도', '증시기상도', \
        '마감시황', '주식부자 속출', '티타임 공략주', '\d{1,2}일 증권사 추천종목', '기업IR소식']
제거제목리스트 = '|'.join(제거할기사제목)

## 혹시 내용이 중복되는 경우가 있을 수 있어 중복 제거하고 위의 제거제목리스트에 있는 단어가 하나라도 기사제목에 존재할경우
## 공시나 광고같은 불필요한 뉴스라 간주하고 제거합니다.
뉴스중복제거 = 뉴스_df.drop_duplicates(['뉴스기사본문'])
뉴스 = 뉴스중복제거[~뉴스중복제거['기사제목'].str.contains(제거제목리스트, na=False, case=False)]

## 소괄호나 대괄호안의 단어들은 대부분 기자이름이나 불필요한 단어이므로 먼저 제거하고 한글 및 
## 원활한 토큰화를 위한 띄어쓰기 1번 등을 제외하고 불필요한 문자와 특수문자들을 제거합니다.
pattern1 = r'\([^)]*\)'
pattern2 = r"\[([^]]+)\]"

뉴스['뉴스기사본문전처리'] = [re.sub(pattern1, '', s) for s in 뉴스['뉴스기사본문']]
뉴스['뉴스기사본문전처리'] = [re.sub(pattern2, '', s) for s in 뉴스['뉴스기사본문전처리']]
뉴스['뉴스기사본문전처리'] = [re.sub('[^/^$|\s+/가-힣\.]', '', s) for s in 뉴스['뉴스기사본문전처리']]
뉴스['뉴스기사본문전처리'] = [re.sub('[[ \s]{2,}\.{2,}]', '', s) for s in 뉴스['뉴스기사본문전처리']]
뉴스['뉴스기사본문전처리'] = [re.sub('\n', '', s) for s in 뉴스['뉴스기사본문전처리']]
뉴스['뉴스기사본문전처리'] = [re.sub('[/+]', '', s) for s in 뉴스['뉴스기사본문전처리']]
뉴스['뉴스기사본문전처리'] = [re.sub('\r', '', s) for s in 뉴스['뉴스기사본문전처리']]
뉴스.drop('뉴스기사본문', axis=1, inplace=True)

mecab = Mecab()
## Mecab으로 명사에 해당하는 단어만 추출합니다.
뉴스['뉴스기사본문전처리'] = 뉴스['뉴스기사본문전처리'].apply(lambda x: mecab.nouns(x))
## 한 글자 단어를 제거합니다.
뉴스['뉴스기사본문전처리'] = 뉴스['뉴스기사본문전처리'].apply(lambda x: [word for word in x if len(word) > 1])

# 토큰화된 뉴스들을 숫자의 시퀀스형태로 변환해줄 모델 불러오고 100글자이내일 경우 100글자까지 패딩합니다.
with open('./model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

뉴스기사본문전처리 = np.array(뉴스['뉴스기사본문전처리'])

뉴스기사본문전처리 = tokenizer.texts_to_sequences(뉴스기사본문전처리)
뉴스기사본문전처리 = sequence.pad_sequences(뉴스기사본문전처리, maxlen=100)

## 뉴스들을 부도 혹은 정상기사로 분류해줄 모델을 불러오고 예측값이 확률로 나오므로 반올림해서 1,0으로 변환합니다. 1의 경우는 부도기사 0은 정상기사 입니다.
model_t = load_model("./model/TomekLinks_Clstm.hdf5")

predict = model_t.predict(뉴스기사본문전처리)
predict = predict.round()

## 뉴스 데이터프레임에 예측값을 부도기사분류라는 이름으로 붙이고 groupby를 활용하여 부도기사비율을 계산합니다.
뉴스['부도기사분류'] = predict
전체기사수 = pd.DataFrame(뉴스.groupby(['기업']).count()['부도기사분류'])
전체기사수.reset_index(drop = False, inplace = True)
전체기사수.columns = ['회사명', '전체기사수']

부도기사수 = pd.DataFrame(뉴스[뉴스.부도기사분류==1].groupby(['기업']).count()['부도기사분류'])
부도기사수.reset_index(drop = False, inplace = True)
부도기사수.columns = ['회사명', '부도기사수']

부도기사비율 = pd.merge(전체기사수, 부도기사수, on='회사명')
부도기사비율['부도기사비율'] = (부도기사비율['부도기사수'] / 부도기사비율['전체기사수']) * 100
부도기사비율.drop(['전체기사수', '부도기사수'], axis=1, inplace=True)

## 특정 기업의 경우 기사가 없어 부도기사비율이 존재하지 않을 수 있어 부도예측을 하고자하는 회사의 목록이 다 있는 사용할재무비율 데이터를 기준으로
## left조인을 합니다. 부도기사가 없어 na값인 경우는 0으로 대체합니다.
사용할재무비율_최종  = pd.merge(사용할재무비율, 부도기사비율, on='회사명', how='left')
사용할재무비율_최종['부도기사비율'].fillna(0, inplace=True)

## 부도기사 여부를 분류할 모델과 독립변수들의 이상치를 처리하기 위해 RobustScaler를 불러옵니다.
ml_model = joblib.load('./model/svc_model.pkl')
robscaler = joblib.load('./model/robustscaler.pkl')

## 독립변수들을 정규화하고 예측
사용할재무비율_최종.iloc[:,2:] = robscaler.transform(사용할재무비율_최종.iloc[:,2:])
사용할재무비율_최종.iloc[:,2:]
predict = ml_model.predict(사용할재무비율_최종.iloc[:,2:])

## 최종결과를 출력하기 위해 데이터 프레임 생성
# 결과_df = pd.DataFrame(columns=['회사명', '부도예측결과'])
# 결과_df['회사명'] = 사용할재무비율_최종['회사명']
# 결과_df['부도예측결과'] = predict

사용할재무비율_최종['부도예측결과'] = predict

for 회사, 결과 in zip(사용할재무비율_최종['회사명'], 사용할재무비율_최종['부도예측결과']):
    if 결과 == 1:
        print(f'{회사}는 1년이내에 상장폐지(부도)될 확률이 높다고 판단됩니다.')
    
    else:
        print(f'{회사}는 1년이내에 상장폐지(부도)될 확률이 낮다고 판단됩니다.')

for i in np.arange(len(사용할재무비율_최종["부도예측결과"])):
    if 사용할재무비율_최종["부도예측결과"][i] == 1:
        사용할재무비율_최종["부도예측결과"][i] = '부도'
    else:
        사용할재무비율_최종["부도예측결과"] = '정상' 

사용할재무비율_최종.to_csv('예측결과.csv', index=None, encoding='utf-8')
