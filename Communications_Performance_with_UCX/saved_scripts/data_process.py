import argparse
import asyncio
from collections import defaultdict
from typing import DefaultDict

from numpy import Inf

DIR = "./job-out/"

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        help='total number of GPU devices.',
    )
    parser.add_argument(
        "-t",
        "--iter", 
        type=int,
        help='total number of iterations'
    )
    args = parser.parse_args()
    return args

def get_info(id, bd_rec):
    file = open(DIR + "worker-" + str(id) + ".out")
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
    

def save_rec(rank1, rank2, event, side, length, t, tot_rec):
    print(rank1, rank2, event, side, length, t)
    tot_rec.append({
                "rank1": int(rank1),
                "rank2": int(rank2),
                "event": event,
                "side": side,
                "length": length,
                "time": t
            })

def output_rec(rec, tot_rec):
    save_rec(rec["rank1"], rec["rank2"], rec["event"], rec["side"], rec["length"], rec["time"], tot_rec)

def get_event(event, bd_rec, tot_rec):
    for rec in bd_rec:
        if(rec["event"] == event):
            output_rec(rec, tot_rec)
        
def get_pair_event(event, bd_rec, tot_rec):
    s_event = "s" + event
    r_event = "r" + event
    lastid = [defaultdict(lambda: -1) for i in range(16)]
    for rec in bd_rec:
        if(rec["event"] == s_event):
            rank1 = rec["rank1"]
            rank2 = rec["rank2"]
            st = rec["time"]
            side = rec["side"]
            mlen = rec["length"]
            for id, rec_p in enumerate(bd_rec):
                if(id <= lastid[rank1][rank2]):
                    continue
                if(rec_p["event"] == r_event and rec_p["side"] == side and rec_p["rank1"] == rank2 and rec_p["rank2"] == rank1):
                    rt = rec_p["time"]
                    save_rec(rank1, rank2, event, side, rec["length"], rt - st, tot_rec)
                    lastid[rank1][rank2] = id
                    break

INF = 9000000000
def get_total_event(event, bd_rec, tot_rec):
    s_event = "s" + event
    r_event = "r" + event
    start = INF
    end = 0
    totlen = 0
    num = 0

    for rec in bd_rec:
        if(rec["event"] == s_event):
            output_rec(rec, tot_rec)
            start = min(start, rec["time"])
            totlen += rec["length"]
            num += 1
            for rec_p in bd_rec:
                if(rec_p["event"] == r_event and rec_p["rank1"] == rec["rank1"]):
                    output_rec(rec_p, tot_rec)
                    end = max(end, rec_p["time"])
                    save_rec(rec["rank1"], rec["rank2"], "total", rec["side"], rec["length"], rec_p["time"] - rec["time"], tot_rec)
                    break
    if(num != 0):
        save_rec(-1, -1, "total_max", "l", totlen / num, end - start, tot_rec)

def process_iter(id, bd_rec, tot_rec):
    del_event = ["sheadp", "rheadp", "rframep", "merge"]
    print("--------------------------", " iter=" + str(id), "--------------------------")
    t_event = ["total"]
    for event in t_event:
        get_total_event(event, bd_rec, tot_rec)

    for event in del_event:
        get_event(event, bd_rec, tot_rec)

    s_event = ["head", "frame"]
    for event in s_event:
        get_pair_event(event, bd_rec, tot_rec)

def get_all_iter_event(event, tot_rec):
    mark = defaultdict(lambda: 0)
    for rec in tot_rec:
        rank1 = rec["rank1"]
        rank2 = rec["rank2"]
        hashstr = str(rank1) + "|" + str(rank2) + event
        if(mark[hashstr] == 0 and rec["event"] == event):
            mark[hashstr] = 1
            tot_len = 0
            tot_t = 0
            tot_num = 0
            for rec_p in tot_rec:
                if(rec_p["rank1"] == rank1 and rec_p["rank2"] == rank2 and rec_p["event"] == event):
                    tot_num += 1
                    tot_len += rec_p["length"]
                    tot_t += rec_p["time"]
            if(tot_num != 0):
                tot_len /= tot_num
                tot_t /= tot_num
                if(event != "frame" and not event.startswith("tot")):
                    print(rank1, rank2, event, "l", tot_len, tot_t)
                else:
                    print(rank1, rank2, event, "l", tot_len, tot_t, tot_len / tot_t / 1024 / 1024)
            

def process_tot(tot_rec):
    print("--------------------------", "total", "--------------------------")
    events = ["total", "total_max", "sheadp", "rheadp", "rframep", "merge", "head", "frame"]
    for event in events:
        get_all_iter_event(event, tot_rec)

def main():
    args = parse_args()
    bd_rec = [[] for i in range(args.iter)]
    tot_rec = []
    for i in range(args.device): 
        get_info(i, bd_rec)
        # get_worker_info(i, bd_rec)
    for i in range(args.iter):
        process_iter(i, bd_rec[i], tot_rec)
    process_tot(tot_rec)

if __name__ == "__main__":
    main()
