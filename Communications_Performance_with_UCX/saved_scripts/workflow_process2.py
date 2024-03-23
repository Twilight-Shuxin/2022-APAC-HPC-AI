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
    events = ['shead', 'shead_1', 'sframe', 'sframe_1', 'rhead', 'rhead_1', 'rframe', 'rframe_1']
    bd_rec = []
    start = INF
    for rec in bd_rec_raw:
        if(rec['event'] in events):
            bd_rec.append(rec)
            start = min(start, rec['time'])

    tot_bar = (args.dev - 1) * 2
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
        'sframe_l': 'tab:blue',
        'sframe_r': 'tab:purple',
        'rhead': 'tab:red',
        'rframe_l': 'tab:olive',
        'rframe_r': 'tab:green',
    }
    label_colors = {
        'shead': 's_header',
        'sframe_l': 's_frame(l)',
        'sframe_r': 's_frame(r)',
        'rhead': 'r_header',
        'rframe_l': 'r_frame(l)',
        'rframe_r': 'r_frame(r)',
    }
    for rec in bd_rec:
        rank = rec['rank2']
        event = rec['event']
        side = rec['side']
        time_spot = rec['time'] - start
        mark = int(event.startswith("r"))
        if(event.endswith("_1")):
            continue
        # print(rank, event, time_spot)
        bar_width = 2
        if(event.endswith("frame")):
            color_event_name = event + "_" + side
        else: 
            color_event_name = event
        color = color_set[color_event_name]
        labeling = '_'
        if(label_colors[color_event_name] != ''):
            labeling = label_colors[color_event_name]
            label_colors[color_event_name] = ''
        # color =  tuple(color_x / 255 for color_x in color)
        for rec_p in bd_rec:
            event_p = rec_p['event']
            rank_p = rec_p['rank2']
            side_p = rec_p['side']
            time_spot_p = rec_p['time'] - start
            if((event_p == event + "_1") and side_p == side and rank_p == rank):
                rank_id = rank * 2 - (2 if (rank > id) else 0)
                bar_id = rank_id + (1 if mark == 0 else 0)
                X_bar[bar_id].append((time_spot, time_spot_p - time_spot))
                Y_bar[bar_id] = ((4 * rank - (4 if (rank > id) else 0) + 1 + \
                    (1 if mark == 0 else 0)) * bar_width, bar_width)
                Y_ticks[bar_id] = (4 * rank - (4 if (rank > id) else 0) + 1 + \
                   (1 if mark == 0 else 0)) * bar_width + (bar_width // 2)
                Y_tick_labels[bar_id] = (str(rank) + ": " + ("recv" if mark else "send") + "_" + side)
                C[bar_id] = C[bar_id] + (color, )
                labels[bar_id] += (labeling, )
    print(X_bar)
    print(labels)
    fig, ax = plt.subplots()
    # fig.set_figheight(20)
    # fig.set_figwidth(7)
    ax.set_ylim(0, (1 + 4 * (args.dev - 1)) * bar_width)
    for bar_id in range(tot_bar):
        for bar in range(len(X_bar[bar_id])):
            ax.broken_barh([X_bar[bar_id][bar]], Y_bar[bar_id], facecolors = C[bar_id][bar], label = labels[bar_id][bar])
    ax.set_yticks(Y_ticks, Y_tick_labels)
    plt.legend(ncol = 3, handlelength = 0.7)
    plt.savefig('workflow_' + str(id) + "_3.png", bbox_inches = 'tight')
    return

def main():
    args = parser.parse_args()
    for i in range(args.k):
        bd_rec = [[] for i in range(args.iter)]
        get_info(i, bd_rec)
        profile_worker(i, bd_rec[args.iter - 1], args)

if __name__ == "__main__":
    main()