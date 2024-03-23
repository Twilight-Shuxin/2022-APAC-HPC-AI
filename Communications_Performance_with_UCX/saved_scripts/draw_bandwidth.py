import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    help='Name of input file.',
)
parser.add_argument(
    "-d",
    "--device",
    type=int,
    help="Total number of input devices",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help='Name of output file',
)
parser.add_argument(
    "-p",
    "--pic1",
    type=str,
    help='Name of output picture1',
)
parser.add_argument(
    "-g",
    "--graph2",
    type=str,
    help='Name of output picture2',
)
args = parser.parse_args()

infile = open(args.input)
lines = infile.readlines()
infile.close()
ngpu = args.device

id = 0
peer = 0
bd = [[0 for j in range(ngpu)] for i in range(ngpu)]
bd_node = [[0 for j in range(ngpu // 4)] for i in range(ngpu // 4)]
tlen = [[0 for j in range(ngpu // 4)] for i in range(ngpu // 4)]
rec = [[0 for j in range(ngpu // 4)] for i in range(ngpu // 4)]

for line in lines:
    data = line.split()
    p1 = int(data[1])
    p2 = int(data[2])
    bd[p1][p2] = float(data[5])
    node1 = p1 // 4
    node2 = p2 // 4
    bd_node[node1][node2] += float(data[5])
    tlen[node1][node2] += float(data[3])
    rec[node1][node2] += 1

outfile = open(args.output, "w")
for i in range(ngpu // 4):
    for j in range(ngpu // 4):
        if(rec[i][j] > 0):
            bd_node[i][j] /= rec[i][j]
            tlen[i][j] /= rec[i][j]
        outfile.write(str(i) + " " + str(j) + " " + str(tlen[i][j]) + " " + str(bd_node[i][j]) + "\n")
outfile.close()
# print(same_tot / same_cnt, diff_tot / diff_cnt)

uniform_data = np.asarray(bd)
fig1, ax1 = plt.subplots()
ax1 = sns.heatmap(uniform_data, annot=False, fmt='g', linewidth=0.5)
fig1.savefig(args.pic1)
plt.close(fig1)

uniform_data = np.asarray(bd_node)
fig2, ax2 = plt.subplots()
ax2 = sns.heatmap(uniform_data, annot=True, fmt='g', linewidth=0.5)
fig2.savefig(args.graph2)
plt.close(fig2)