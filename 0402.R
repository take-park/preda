# ========== 4.2 1교시 : β 줄이기(변수선택) ==========

# 입력 변수의 수를 줄였을때의 장점
# - 잡음(noise)를 제거해 모형의 정확도를 개선함
# - 모형의 연산 속도가 빨라짐
# - 다중공선성의 문제를 제거해 모형의 해석 능력을 향상시킴
# - 예를들어 입력 변수가 나이, 생년이 있는경우 둘은 같은 의미를 갖기 때문에 하나를 제거함 
# - 계수축소법에는 Ridge와 LASSO 방법이 있음

# ===== 1.1] Ridge(능선) regression
# Rigde 회귀에서는 f(beta)에 회귀계수의 제곱의 합을 대입함
# - L2 norm 으로 표현
# - λ(람다)는 tuning parameter로 클수록 
# - 많은 회귀계수를 0으로 수렴시킴
## f(a,b;x,y) = { Y - X*β }^2  (X = 1더해진 x메트릭스)
##            = (Y-Xβ)^T * (Y-Xβ) + λ*β^2
## hat β(Ridge) = (X^T * X + λ*I)^{-1} * X^T * Y =====

# ===== 1.2] LASSO regression
# - (Least Absolute Shrinkage and Selection Operator) 
# - LASSO 회귀에서는 f(beta)에 회귀계수의 절대값의 합을 대입함
# - λ(람다)는 tuning parameter로 클수록 
# - 많은 회귀계수를 0으로 수렴시킴

# 공통점
# - 계수축소법으로 잔차와 회귀계수를 최소화하는 최적화 문제임
# - 즉 min SSE + f(beta)의 목적함수를 푸는 문제임

# 차이점
# - Ridge는 계수가 0에 가깝게 축소되는데 비해 
# - LASSO는 계수가 0으로 축소됨
# - Rigde는 입력 변수들이 전반적으로 비슷한 수준으로 
# - 출력 변수에 영향을 미치는 경우에 사용함
# - LASSO는 출력 변수에 미치는 
# - 입력 변수의 영향력 편차가 큰 경우에 사용함

# 출처: https://neocarus.tistory.com/entry/Ridge-regression과-LASSO-regression [Passive Incomer's Active Life]


