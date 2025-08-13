#code:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptro
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
    prediction = clf.predict([X[i]])[0]
    actual = y[i]
    remark = "Correct" if prediction == actual else "May fail"
    print(f"Input {X[i]} => Predicted Output: {prediction} (Expected: {actual}) - {remark}")
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red')  # Actual output 0
    else:
        plt.scatter(X[i][0], X[i][1], color='blue')  # Actual output 
x_values = np.array([0, 1])
y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]  # Perceptron decision boundary
plt.plot(x_values, y_values, label="Decision Boundary", color='green')
plt.title('Perceptron Decision Boundary for XOR')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.legend()
plt.show()

#Output with code(screenshot):<img width="762" height="649" alt="Screenshot 2025-08-13 102903" src="https://github.com/user-attachments/assets/645cf65d-05c3-43c4-9cc9-3adb63927bbb" />
<img width="748" height="484" alt="Screenshot 2025-08-13 102914" src="https://github.com/user-attachments/assets/c0f80b2e-c40e-42f5-8eae-fe7d32631464" />
