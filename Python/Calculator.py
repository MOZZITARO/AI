from Calculator_operations import Mathoperations

class Calculator:
     def __init__(self):
         self.math_operations = Mathoperations()
         
     def perform_operations(self, a, b):
         print(f"Addition of {a} and {b}: {self.math.operations.add(a,b)}")
         print(f"Subtration of {a} and {b}: {self.math_operations.subtract(a, b)}")
         print(f"Multiplication of {a} and {b}: {self.math_operations.multiply(a, b)}")
         print(f"Division of {a} and {b}: {self.math_operations.divide(a, b)}")
         
    #실행 예제
     if __name__ == "__main__":
         Calculator = Calculator()
         Calculator.perform_operations(10, 5)
         
         