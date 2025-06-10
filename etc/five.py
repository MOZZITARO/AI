# 컬럼 설명 : satisfaction_level(직원만족도점수), last_evaluation(고용주평가점수), numbers_projects(할당된 프로젝트수)
# average_monthly_hours(한달동안 직원이 일한 평균시간), time_spent_company(회사에서 근무한 연수), work_accident(근무중 사고유무무)
# promotion_last_5years(지난 5년 직원이 승진했는지 여부), Departments(부서), Salary(월급수준 낮음, 중간, 높음), left(직원퇴사 여부)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# - 데이터셋: 'HR_comma_sep.csv' 파일을 사용합니다.
# - 주요 처리: 'Departments' 열 이름 수정 (공백 제거)
# 범주형 변수('Departments', 'salary')를 One-Hot Encoding으로 변환
# drop_first=True 옵션을 사용하여 다중공선성 문제 방지
data = pd.read_csv('dataset/HR_comma_sep.csv')
data.head()


# 데이터 지정보다 먼저
# 데이터 처리
# 열 이름 수정
data.rename(columns={'Departments ': 'Departments'}, inplace=True)
# 원 샷 핫코딩
data = pd.get_dummies(data, columns=['Departments', 'salary'], drop_first=True)


selected_features = ['satisfaction_level', 'number_project', 'time_spend_company']
X = data[selected_features]
y = data['left']


# 데이터 분할 (Train: 80%, Test: 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)

# 랜덤포레스트에는 test_size없음
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# "모델 저장" (피클통)
joblib.dump(model, 'company_model.pkl')

# 예측
y_pred = model.predict(X_test)

# 정확도 측정 
accuracy = accuracy_score(y_test , y_pred)
print(classification_report(y_test, y_pred))
print(f"Model accuracy: {accuracy * 100:.2f}")

importance = pd.DataFrame ({
    
        '특성' : selected_features,
        '중요성': model.feature_importances_
    
}).sort_values('중요성' , ascending=False)

st.bar_chart(importance.set_index('특성'))



# UI 구현
satisfaction_level = st.slider('만족도', min_value=0.00, max_value=1.00)
number_project = st.number_input('프로젝트 수', min_value=1, max_value=10)
#st.number_input(label, min_value=None, max_value=None, value=default, step=1, format=None, key=None)
time_spend_company = st.number_input('근속 년수', min_value=1, max_value=20)

if st.button("예측하기"):
   model = joblib.load('company_model.pkl')
   input_data = np.array([[satisfaction_level, number_project, time_spend_company]])
   prediction = model.predict(input_data)[0]
   proba = model.predict_proba(input_data)[0]
   #결과 출력
   if prediction == 1:
        st.error(f"직원 퇴사 확률이 높습니다 , {proba[1] * 100}%")
   else:
        st.success(f"직원 퇴사 확률이 낮습니다 , {proba[0] * 100}%")

