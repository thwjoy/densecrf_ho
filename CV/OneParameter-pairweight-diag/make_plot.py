import matplotlib.pyplot as plt


scores5 = []
params5 = []

scores15 = []
params15 = []

scores25 = []
params25 = []

for score, param in perfs_525_5:
    scores5.append(-score)
    params5.append(param)

for score, param in perfs_525_15:
    scores15.append(-score)
    params15.append(param)

for score, param in perfs_525_25:
    scores25.append(-score)
    params25.append(param)


plt.plot(params5, scores5, 'ro', label= "5 iterations")
plt.plot(params15, scores15, 'bo', label = "15 iterations")
plt.plot(params25, scores25, 'go', label = "25 iterations")
plt.legend()
plt.show()
