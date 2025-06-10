import pymysql
from pymysql import Error

class Database:
    def __init__(self):
        self.connection = None
        try:
            self.connection = pymysql.connect(
                host='localhost',
                database='test',  # test 데이터베이스 사용
                user='root',
                password='kirito1013@',  # mariadb 설치 당시의 패스워드, 실제 환경에서는 보안을 위해 환경변수 등을 사용
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("MariaDB에 성공적으로 연결되었습니다.")
        except Error as e:
            print(f"MariaDB 연결 중 오류 발생: {e}")

    # def save_bmi_record(self, weight, height, bmi, category):
    #     """BMI 기록을 데이터베이스에 저장"""
    #     try:
    #         if self.connection is None:
    #             print("데이터베이스 연결이 없습니다.")
    #             return False
                
    #         with self.connection.cursor() as cursor:
    #             query = """
    #             INSERT INTO bmi_records (weight, height, bmi, category)
    #             VALUES (%s, %s, %s, %s)
    #             """
    #             cursor.execute(query, (weight, height, bmi, category))
            
    #         self.connection.commit()
    #         print("BMI 기록이 성공적으로 저장되었습니다.")
    #         return True
    #     except Error as e:
    #         print(f"데이터 저장 중 오류 발생: {e}")
    #         return False
    
    def insert_foodcal_records(self, str, allcal):
        '''입력한 음식(들)과 총 칼로리와 날짜를 저장'''
        try:
            if self.connection is None:
                print("Db connecion is None")
                return False
            
            with self.connection.cursor() as cursor:
                query = "insert into foodcal_records (allfood, allcalories) values (%s, %s)"
                cursor.execute(query, (str, allcal))
            self.connection.commit()
            print("기록이 저장되었습니다.")
            return True
        
        except Error as e:
            print(f"데이터 저장 중 오류 발생: {e}")
            return False        
    
    
    # def get_bmi_records(self, limit=10):
    #     """최근 BMI 기록을 가져옵니다"""
    #     try:
    #         if self.connection is None:
    #             print("데이터베이스 연결이 없습니다.")
    #             return []
                
    #         with self.connection.cursor() as cursor:
    #             query = """
    #             SELECT * FROM bmi_records
    #             ORDER BY created_at DESC
    #             LIMIT %s
    #             """
    #             cursor.execute(query, (limit,))
    #             records = cursor.fetchall()
            
    #         return records
    #     except Error as e:
    #         print(f"데이터 조회 중 오류 발생: {e}")
    #         return []
    
    # def get_all_food_calories(self, List):
    #     '''입력한 음식들의 칼로리를 가져와서 총 칼로리 계산'''
    #     try:
    #         if self.connection is None:
    #             print("Db Connection is none")
    #             return []
            
    #         with self.connection.cursor() as cursor:
                
    #             # calList = []  # 조회된 칼로리를 저장할 리스트 선언
    #             sum = 0     # 총 칼로리
    #             for i in range(len(List)):
    #                 query = "select cal from foodcal where food = %s"
    #                 cursor.execute(query, (List[i],))
    #                 foodcal = cursor.fetchone()   # 입력한 음식의 칼로리 가져오기
    #                 print(foodcal)
    #                 #calList.append(foodcal)
    #                 sum += foodcal.get('cal',0)
    #                 print(sum)
                
    #             return sum
    #     except Error as e:
    #         print(f"데이터 조회 중 오류 발생: {e}")
    #         return []
    
    def get_all_food_calories(self, food_list):
        '''입력한 음식들의 칼로리를 가져와서 총 칼로리 계산'''
        try:
            if self.connection is None:
                print("Db Connection is none")
                return 0
        
            with self.connection.cursor() as cursor:
                total_cal = 0
                for food in food_list:
                    food = food.strip()
                    query = "SELECT cal FROM foodcal WHERE food = %s"
                    cursor.execute(query, (food,))
                    foodcal = cursor.fetchone()

                    # None이면 .get() 하지 않고 넘어감
                    if foodcal is not None:
                        total_cal += foodcal.get('cal', 0)
                    else:
                        print(f"'{food}'에 대한 칼로리 정보를 찾을 수 없습니다.")

                return total_cal
        except Error as e:
            print(f"데이터 조회 중 오류 발생: {e}")
            return 0
    
    def get_exercise_cal(self):
        '''운동 종류와 소모 칼로리 가져오기'''
        try:
            if self.connection is None:
                print("Db Connection is none")
                return []
            
            with self.connection.cursor() as cursor:
                
                query = "select exercise, exercise_cal from exercise"
                cursor.execute(query)
                exercises = cursor.fetchall()   # 운동 종류와 소모 칼로리 가져오기
        
                return exercises
        except Error as e:
            print(f"데이터 조회 중 오류 발생: {e}")
            return []        
                    

    def get_records(self, limit=10):
        '''기록 가져오기'''
        try:
            if self.connection is None:
                print("Db Connection is none")
                return []
            
            with self.connection.cursor() as cursor:
                
                query = "select * from foodcal_records order by created_at desc LIMIT %s"
                 
                cursor.execute(query, (limit,))
                records = cursor.fetchall()   # 모든 기록 가져오기
        
                return records
        except Error as e:
            print(f"데이터 조회 중 오류 발생: {e}")
            return []    

    def get_foods(self):
        '''음식 목록 가져오기'''
        try:
            if self.connection is None:
                print("Db Connection is none")
                return []
            
            with self.connection.cursor() as cursor:
                
                query = "select * from foodcal"
                 
                cursor.execute(query)
                foodlist = cursor.fetchall()   # 모든 기록 가져오기
        
                return foodlist
        except Error as e:
            print(f"데이터 조회 중 오류 발생: {e}")
            return []  
        

    def close(self):
        """데이터베이스 연결 종료"""
        if self.connection:
            self.connection.close()
            print("MariaDB 연결이 종료되었습니다.")