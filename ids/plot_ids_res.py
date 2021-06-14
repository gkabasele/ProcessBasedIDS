import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ["Detected", "#False alert", "TPR", "FPR", "DTime(s)", "IFP-Time(s)"]
x = np.arange(len(labels))
width = 0.20

# HDBSCAN for every transition 
#timepattern = [round(17/17, 2), 16, 0.77, 0.008, 5, 76]
#Using only Local Outlier Factor in detection
#Medium process
medium_tp = [round(17/17, 2), 16, 0.78, 0.01, 5, 78]
medium_tp_error = [0, 0, 0, 0, 8.59, 114.34]
medium_ar = [round(10/17, 2), 161, 0.25, 0.08, 2.9, 10]
medium_ar_error = [0, 0, 0, 0, 3.11, 43.59]
medium_inv = [round(15/17, 2), 48, 0.45, 0.026, 2.5, 31]
medium_inv_error = [0, 0, 0, 0, 2.39, 142.5]

#SWAT Process
swat_tp = [round(29/36, 2), 11494, 0.83, 0.02, 124.65, 39.01]
swat_tp_error = [0, 0, 0, 0, 155.5, 422.55]
swat_ar = [round(36/36, 2), 56718, 0.17, 0.12, 10.08, 7.93]
swat_ar_error = [0, 0, 0, 0, 19.27, 152.39]
swat_inv = [round(17/36, 2), 1276, 0.64, 0.002, 307.24, 347.56]
swat_inv_error = [0, 0, 0, 0, 594.54, 1967.33]

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{}".format(height),
                  xy=(rect.get_x() + rect.get_width()/2, height),
                  xytext=(0,3),
                  textcoords="offset points",
                  ha="center", va="bottom")

def plot_ids_comparison(title, ids1, label1, error1, ids2, label2, error2,
                        ids3, label3, error3, graph_labels):

    x = np.arange(len(graph_labels))
    width = 0.20

    fig, ax = plt.subplots()
    
    rects1 = ax.bar(x - width, ids1, width, label=label1, yerr=error1)
    rects2 = ax.bar(x, ids2, width, label=label2, yerr=error2)
    rects3 = ax.bar(x + width, ids3, width, label=label3, yerr=error3)

    ax.set_title(title)
    
    ax.set_ylabel("Values")
    ax.set_xticks(x)
    ax.set_xticklabels(graph_labels)
    ax.legend()

    plt.yscale("log")
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    plt.show()

plot_ids_comparison("SWaT Process",swat_tp, "TimePattern", swat_tp_error, swat_ar,
                    "AR", swat_ar_error, swat_inv, "Invariant",
                    swat_inv_error, labels)

plot_ids_comparison("Medium Process", medium_tp, "TimePattern", medium_tp_error, medium_ar,
                    "AR", medium_ar_error, medium_inv, "Invariant",
                    swat_inv_error, labels)
