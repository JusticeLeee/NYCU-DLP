import matplotlib.pyplot as plt

import numpy as np
score = np.zeros(500000)
f = open("score.txt", "r")
for i in range (500000):
    score[i] = f.readline()
f.close

plt.plot(score)
plt.xlabel("epoch")
plt.ylabel("score")
plt.show()