import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["Detected", "#False alert", "TPR", "FPR", "DTime(s)", "IFP-Time(s)"]
x = np.arange(len(labels))
width = 0.20

timepattern = [round(17/17, 2), 16, 0.77, 0.008, 5, 76]
ar = [round(10/17, 2), 161, 0.25, 0.08, 2.9, 10]
invariant = [round(15/17, 2), 48, 0.45, 0.026, 2.5, 31]

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, timepattern, width, label="TimePat")
rects2 = ax.bar(x , ar, width, label="AR")
rects3 = ax.bar(x + width, invariant, width, label="Inv")

ax.set_ylabel("Values")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{}".format(height),
                  xy=(rect.get_x() + rect.get_width()/2, height),
                  xytext=(0,3),
                  textcoords="offset points",
                  ha="center", va="bottom")

plt.yscale("log")
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()
