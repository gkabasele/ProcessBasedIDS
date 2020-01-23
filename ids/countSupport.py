import argparse

def main(dbfile, supfile, statfile, gamma, theta):
    support_map = {}
    trans_size = 0
    n_trans = 0
    avg_trans = 0
    max_trans = 0
    on_all_trans = 0
    with open(dbfile, 'r') as db:
        for line in db:
            items = line.split()
            trans_size += len(items)
            max_trans = max(max_trans, len(items))
            for i in items:
                if i not in support_map:
                    support_map[i] = 1
                else:
                    support_map[i] += 1
            n_trans += 1
        avg_trans = float(trans_size)/n_trans

    num_freq = 0
    mis_theta = int(theta * n_trans)
    with open(supfile, 'w') as supports:
        for k, v in support_map.items():
            if v >= theta:
                num_freq += 1
            if v == n_trans:
                on_all_trans += 1
            sup = max(int(gamma*v), mis_theta)
            supports.write("{} : {}\n".format(k, sup))

    with open(statfile, 'w') as stat:
        stat.write("Dataset Statistic\n")
        stat.write("Nbr Trans.: {}\n".format(n_trans))
        stat.write("Max Trans. Size: {}\n".format(max_trans))
        stat.write("Avg Trans. Size: {}\n".format(avg_trans))
        stat.write("Number of items with a support greater than {} :{}\n".format(mis_theta,
                                                                                num_freq))
        stat.write("Number of items appearing all the time: {}\n".format(on_all_trans))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="store", dest="dbfile")
    parser.add_argument("--sup", action="store", dest="supfile")
    parser.add_argument("--stat", action="store", dest="statfile")
    parser.add_argument("--gamma", action="store", type=float,
                        default=0.9, dest="gamma")
    parser.add_argument("--theta", action="store", type=float,
                        default=0.32, dest="theta")

    args = parser.parse_args()

    main(args.dbfile, args.supfile, args.statfile, args.gamma, args.theta)
