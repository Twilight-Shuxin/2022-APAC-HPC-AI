import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description = "Get workflow graph for workers with id 0 ~ k-1")
parser.add_argument("-d", "--dev", type = int, help = "total number of devices involved")
parser.add_argument("-k", help = "get id range for worker to plot graph", type = int)
parser.add_argument("-t", "--iter", type = int, help = "total number of iterations")

DIR = "./job-out/"

def get_info(id, bd_rec):
    file = open(DIR + "worker-" + str(id) + ".out")
    #file = open(DIR + "gadi-worker.out")
    lines = file.readlines()
    for line in lines:
        input = line.split(" ")
        iter, rank1, rank2, event, side, length, t = input
        iter = int(iter)
        rank1 = int(rank1)
        rank2 = int(rank2)
        length = int(float(length))
        t = float(t)
        bd_rec[iter].append(
            {
                "rank1": rank1,
                "rank2": rank2,
                "event": event,
                "side": side,
                "length": length,
                "time": t
            }
        )

INF = 9000000000

def profile_worker(id, bd_rec_raw, args):
    events = ['sframe', 'rframe', 'shead', 'rhead', 'sframe_1', 'rhead_0']
    event_pair = {
        'shead': 'sframe',
        'sframe': 'sframe_1',
        'rhead_0': 'rhead',
        'rhead': 'rframe'
    }
    ending_only = ['sframe_1', 'rframe']
    bd_rec = []
    start = INF
    for rec in bd_rec_raw:
        if(rec['event'] in events):
            bd_rec.append(rec)
            start = min(start, rec['time'])
    tot_bar = (args.dev - 1) * 4
    X_bar = [[] for _ in range(tot_bar)]
    Y_bar = [(0, 0) for _ in range(tot_bar)]
    Y_ticks = [0 for _ in range(tot_bar)]
    Y_tick_labels = [0 for _ in range(tot_bar)]
    C = [() for _ in range(tot_bar)]
    labels = [() for _ in range(tot_bar)]
    # color_set = {
    #     'shead': (44, 113, 200),
    #     'sframe': (233, 249, 157),
    #     'rhead': (252, 63, 0),
    #     'rhead_0': (252, 182, 171),
    # }
    color_set = {
        'shead': 'tab:orange',
        'sframe': 'tab:red',
        'rhead': 'tab:blue',
        'rhead_0': 'tab:green',
    }
    rem_colors = {
        'shead': 's_header',
        'sframe': 's_frame',
        'rhead': 'r_frame',
        'rhead_0': 'r_header'
    }
    for rec in bd_rec:
        rank = rec['rank2']
        event = rec['event']
        side = rec['side']
        time_spot = rec['time'] - start
        mark = int(event.startswith("r"))
        if(event in ending_only):
            continue
        # print(rank, event, time_spot)
        bar_width = 2
        color = color_set[event]
        labeling = '_'
        if(rem_colors[event] != ''):
            labeling = rem_colors[event]
            rem_colors[event] = ''
        # color =  tuple(color_x / 255 for color_x in color)
        for rec_p in bd_rec:
            event_p = rec_p['event']
            rank_p = rec_p['rank2']
            side_p = rec_p['side']
            time_spot_p = rec_p['time'] - start
            if(event_p == event_pair[event] and side_p == side and rank_p == rank):
                rank_id = rank * 4 - (4 if (rank > id) else 0)
                bar_id = rank_id + (2 if side == 'l' else 0) + (1 if mark == 0 else 0)
                print(id, rank, side, rank_id, mark, event, bar_id)
                X_bar[bar_id].append((time_spot, time_spot_p - time_spot))
                Y_bar[bar_id] = ((6 * rank - (6 if (rank > id) else 0) + 1 + (2 if side == 'l' else 0) + \
                    (1 if mark == 0 else 0)) * bar_width, bar_width)
                Y_ticks[bar_id] = (6 * rank - (6 if (rank > id) else 0) + 1 + (2 if side == 'l' else 0) + \
                   (1 if mark == 0 else 0)) * bar_width + (bar_width // 2)
                Y_tick_labels[bar_id] = (str(rank) + ": " + ("recv" if mark else "send") + "_" + side)
                C[bar_id] = C[bar_id] + (color, )
                labels[bar_id] += (labeling, )
    print(X_bar)
    print(labels)
    fig, ax = plt.subplots()
    # fig.set_figheight(20)
    # fig.set_figwidth(7)
    ax.set_ylim(0, (1 + 6 * (args.dev - 1)) * bar_width)
    for bar_id in range(tot_bar):
       ax.broken_barh([X_bar[bar_id][0]], Y_bar[bar_id], facecolors = C[bar_id][0], label = labels[bar_id][0])
       ax.broken_barh([X_bar[bar_id][1]], Y_bar[bar_id], facecolors = C[bar_id][1], label = labels[bar_id][1])
    ax.set_yticks(Y_ticks, Y_tick_labels)
    plt.legend(ncol = 2, handlelength = 0.7)
    plt.savefig('workflow_' + str(id) + "_2.png", bbox_inches = 'tight')
    return

def main():
    args = parser.parse_args()
    for i in range(args.k):
        bd_rec = [[] for i in range(args.iter)]
        get_info(i, bd_rec)
        profile_worker(i, bd_rec[args.iter - 1], args)

if __name__ == "__main__":
    main()