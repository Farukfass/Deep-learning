# Deep-learning
#code:
from keras.models import Sequential
from keras.layers import Dense
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
print("Accuracy:", acc)
print("Predictions:", model.predict(X))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

#Output with code(screeenshot):
<img width="1514" height="617" alt="dl1" src="https://github.com/user-attachments/assets/ebe589e0-099b-49c0-bc23-7f26688eab4b" />
<img width="1492" height="637" alt="dl2" src="https://github.com/user-attachments/assets/9be2baa0-2fa3-4aaf-a3d6-c0e638e5910e" />
