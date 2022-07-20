
#  after running https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html



from statistics import mean

a=regr.coef_
b=regr.intercept_
mse1 = 0
for x, y in zip(diabetes_X_train, diabetes_y_train):
    # print(a*x+b - y)
    mse1 += (a*x+b - y)**2

mse2 = 0
avg = mean(diabetes_y_test)
tot = 0
for x, y in zip(diabetes_X_test, diabetes_y_test):
    # print(a*x+b - y)
    mse2 += (a*x+b - y)**2
    tot += (y- avg)**2

print(f'mse1 {mse1}, mse2 {mse2}')
r2 = 1 - mse2/tot
print(f' r2 {r2}')  # https://en.wikipedia.org/wiki/Coefficient_of_determination