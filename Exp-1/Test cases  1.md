#code:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y)
print(f"Training Accuracy: {acc:.4f}")
predictions = model.predict(X)
predictions_binary = (predictions > 0.5).astype(int)
print("\nTest Case Results:")
print("Input\tExpected\tPredicted\tPass?")
for i in range(len(X)):
    inp = X[i]
    expected = Y[i][0]
    pred = predictions_binary[i][0]
    passed = "Yes" if pred == expected else "No"
    print(f"{inp}\t{expected}\t\t{pred}\t\t{passed}")
plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
<img width="612" height="629" alt="Screenshot 2025-08-13 102817" src="https://github.com/user-attachments/assets/10124b5c-c471-4c2e-9cfb-b5b5c22796af" />
<img width="1542" height="547" alt="Screenshot 2025-08-13 102849" src="https://github.com/user-attachments/assets/2e0fc1fa-abe7-4d5c-a080-0053637d9c5c" />

#output with code(screenshot):
