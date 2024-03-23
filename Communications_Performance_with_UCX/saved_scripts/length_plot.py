from matplotlib import pyplot as plt
import numpy as np


lengths = ["5", "10", "60", "120", "180", "250", "350"]
totprof = {}

for length in lengths:
    file = open("out-ana-" + length + "00000.out")
    lines = file.readlines()
    file.close()
    totprof[length] = {
        "same":[],
        "diff":[],
        "length":[],
    }
    for line in lines:
        nums = line.split()
        prof = {
            "rank1": int(nums[1]),
            "rank2": int(nums[2]),
            "length": float(nums[3]),
            "bandwidth": float(nums[4]),
        }
        if(prof["rank1"] == -1):
            continue
        totprof[length][nums[0]].append(prof["bandwidth"])
        totprof[length]["length"].append(prof["length"])
    for key, value in totprof.copy().items:
        totprof[key] = float(np.mean(np.asarray(value)))


cat = ["bored", "happy", "bored", "bored", "happy", "bored"]
dog = ["happy", "happy", "happy", "happy", "bored", "bored"]
activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

fig, ax = plt.subplots()
ax.plot(activity, dog, label="dog")
ax.plot(activity, cat, label="cat")
ax.legend()

plt.show()