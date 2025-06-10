# 사전 설치 : pip install flask pymysql
from flask import Flask, render_template, request, redirect, url_for
# from bmi import BMICalculator
from db import Database
import atexit   # 애플리케이션 종료시 실행을 요청 (ex. DB연결 종료)

app = Flask(__name__)   # Flask 앱 초기화
db = Database()   # DB 초기화

# 애플리케이션 종료 시 DB 연결 종료
atexit.register(db.close)

@app.route('/', methods=['GET'])
def index():
    foodss = db.get_foods()
    print(foodss)
    return render_template('index_food.html', foods = foodss)

@app.route('/calculate', methods=['POST'])
def calculate():
    
    foods = request.form['foods']    # 입력한 음식
    gender = request.form['gender'] # 입력한 성별
    
    try:

        # weight = float(request.form['weight'])
        # height = float(request.form['height'])
        
        print(foods)
        
        num = int(foods)
        
        return render_template('index_food.html', error="! 숫자가 아닌 한글로 입력해주세요. !")
                
        # result.html로 입력한 음식들, 성별, 총 칼로리, 운동

        # return render_template('result.html', text=foods, gender=gender, allcal=allcal, overcal=overcal,exercise_cal=exercise_cal)
        
        # <p>추가 섭취 필요 칼로리량</p>: <strong>{{}}</strong></p>
        # <p>추천 음식: <strong>{{}}</strong></p>
        
        
        # 입력값 유효성 검사
        # if weight <= 0 or height <= 0:
        #     return render_template('index.html', error="체중과 신장은 양수여야 합니다.")
        
        # BMI 계산
        # calculator = BMICalculator(weight, height)
        # result = calculator.get_result()
        
        # 데이터베이스에 저장
        # db.save_bmi_record(weight, height, result["bmi"], result["category"])
        
        # return render_template('result.html', 
        #                       bmi=result["bmi"], 
        #                       category=result["category"],
        #                       weight=weight,
        #                       height=height)
    except ValueError:
        
        foodList = foods.split(',')   # 입력한 음식들을 ,을 기준으로 쪼개서 리스트에 저장
        print(foodList)
        
        print(gender)
        
            
        # 성인 권장 칼로리
        male_cal = 2800
        female_cal = 2300
        
        # 입력한 음식의 총 칼로리
        allcal = db.get_all_food_calories(foodList)
        print(allcal)
        
        # 기록 테이블에 저장하기
        db.insert_foodcal_records(foods, allcal)
        print("저장 성공")
        
        # 성별이 남자일 때
        if gender == 'male':
            
            # 총 칼로리가 남성 권장 칼로리보다 높으면
            if allcal > male_cal:
                overcal = allcal - male_cal          # 초과되는 칼로리
                print(overcal)
                exercise_cal = db.get_exercise_cal() # 모든 운동 종류의 칼로리(리스트)
                print(exercise_cal)
                
                for cal in exercise_cal:
                    print(f"{overcal}칼로리를 소모하기 위해선 {overcal / cal['exercise_cal']}시간동안 {cal['exercise']}을(를) 해야합니다")
                    # ex: 300칼로리를 소모하기 위해선 3시간동안 달리기를 해야합니다.
                    # ex: 300칼로리를 소모하기 위해선 1시간동안 근력운동을 해야합니다.
                    # ex: 300칼로리를 소모하기 위해선 1시간동안 클라이밍를 해야합니다.
                    
                return render_template('result.html', text=foods, gender=gender, allcal=allcal, overcal=overcal,exercise_cal=exercise_cal)
            else:
                lowcal = male_cal - allcal          # 부족한 칼로리
                print(lowcal)
                foodlist = db.get_foods()           # 음식 목록 가져오기
                print(foodlist)
                
                addFood = []
                # 부족한 칼로리와 같거나 적은 칼로리를 가지는 음식들
                for food in foodlist:
                    if food['CAL'] <= lowcal:
                        addFood.append({
                            'FOOD' : food['FOOD'],
                            'CAL': food['CAL']
                            })
                
                print(addFood)
                
                #for food in foodlist:
                    #print(f"부족한 칼로리를 채우려면 {lowcal / foodlist.cal}개의 {}")
                    
                return render_template('result.html', text=foods, gender=gender, allcal=allcal, lowcal=lowcal, foodlist=addFood)
                    
        # 성별이 여성일 때
        elif gender == 'female':
            
            # 총 칼로리가 여성 권장 칼로리보다 높으면
            if allcal > female_cal:
                overcal = allcal - female_cal          # 초과되는 칼로리
                exercise_cal = db.get_exercise_cal() # 모든 운동 종류의 칼로리(리스트)
            
                #for cal in exercise_cal:
                #    print(f"{overcal}칼로리를 소모하기 위해선 {overcal / cal.exercise_cal}시간동안 {cal.exercise}을(를) 해야합니다")
                    # ex: 300칼로리를 소모하기 위해선 3시간동안 달리기를 해야합니다.
                    # ex: 300칼로리를 소모하기 위해선 1시간동안 근력운동을 해야합니다.
                    # ex: 300칼로리를 소모하기 위해선 1시간동안 클라이밍를 해야합니다.

                return render_template('result.html', text=foods, gender=gender, allcal=allcal, overcal=overcal,exercise_cal=exercise_cal)

            else:
                lowcal = female_cal - allcal          # 부족한 칼로리
                print(lowcal)
                foodlist = db.get_foods()           # 음식 목록 가져오기
                print(foodlist)
                
                addFood = []
                # 부족한 칼로리와 같거나 적은 칼로리를 가지는 음식들
                for food in foodlist:
                    if food['CAL'] <= lowcal:
                        addFood.append({
                            'FOOD' : food['FOOD'],
                            'CAL': food['CAL']
                            })
                
                print(addFood)
                
                #for food in foodlist:
                    #print(f"부족한 칼로리를 채우려면 {lowcal / foodlist.cal}개의 {}")
                    
                return render_template('result.html', text=foods, gender=gender, allcal=allcal, lowcal=lowcal, foodlist=addFood)
        
        
        
        

@app.route('/history')
def history():
    # 최근 BMI 기록 10개 가져오기
    # records = db.get_bmi_records(10)
    # return render_template('history.html', records=records)
    
    # 최근 기록 10개 가져오기
    records = db.get_records(10)
    return render_template('history.html', records = records)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)