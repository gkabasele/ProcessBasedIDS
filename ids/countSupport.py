import argparse
import pdb

def main(dbfile, misfile, supfile, statfile, filterfile, mis_filterfile, gamma, theta):
    support_map = {}
    trans_size = 0
    n_trans = 0
    avg_trans = 0
    max_trans = 0
    on_all_trans = 0
    transactions = []
    with open(dbfile, 'r') as db:
        for line in db:
            items = line.split()
            transactions.append(set(items))
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
    num_infreq = 0
    mis_theta = int(theta * n_trans)
    all_time_items = []
    items = list(support_map.keys())
    items.sort()

    with open(supfile, 'w') as supports:
        with open(misfile, 'w') as mis:
            for k in items:
                v = support_map[k]
                if v >= mis_theta:
                    num_freq += 1
                else:
                    num_infreq += 1
                if v == n_trans:
                    all_time_items.append(k)
                mis_sup = max(int(gamma*v), mis_theta)
                supports.write("{} {}\n".format(k, v))
                mis.write("{} {}\n".format(k, mis_sup))

    with open(statfile, 'w') as stat:
        stat.write("Dataset Statistic\n")
        stat.write("-----------------\n")
        stat.write("Nbr Trans.: {}\n".format(n_trans))
        stat.write("Max Trans. Size: {}\n".format(max_trans))
        stat.write("Avg Trans. Size: {}\n".format(avg_trans))
        stat.write("Number of items with a support greater than {} :{}\n".format(mis_theta,
                                                                                 num_freq))
        stat.write("Number of items with a support less than {} :{}\n".format(mis_theta,
                                                                              num_infreq))

        stat.write("Number of items appearing all the time: {}\n".format(len(all_time_items)))

        all_time_items = set(all_time_items)
        stat.write("{}\n".format(all_time_items))
        trans_size = 0
        avg_trans = 0
        max_trans = 0

        # Filtering the items who appear every time
        with open(filterfile, 'w') as f:
            for t in transactions:
                keep_items = [int(x) for x in (t - all_time_items)]
                keep_items.sort()
                for item in keep_items:
                    f.write("{} ".format(item))
                f.write("\n")
                trans_size += len(keep_items)
                max_trans = max(max_trans, len(keep_items))

            avg_trans = float(trans_size)/n_trans

        with open(mis_filterfile, 'w') as f:
            for k in items:
                if k not in all_time_items:
                    v = support_map[k]
                    mis_sup = max(int(gamma*v), mis_theta)
                    f.write("{} {}\n".format(k, v))

        stat.write("Filter Dataset Statistic\n")
        stat.write("------------------------\n")
        stat.write("Max Trans. Size: {}\n".format(max_trans))
        stat.write("Avg Trans. Size: {}\n".format(avg_trans))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="store", dest="dbfile")
    parser.add_argument("--sup", action="store", dest="supfile")
    parser.add_argument("--mis", action="store", dest="misfile")
    parser.add_argument("--mis_filter", action="store", dest="mis_filter")
    parser.add_argument("--stat", action="store", dest="statfile")
    parser.add_argument("--filter", action="store", dest="filterfile",
                        help="File where to output the database without constant items")
    parser.add_argument("--gamma", action="store", type=float,
                        default=0.9, dest="gamma")
    parser.add_argument("--theta", action="store", type=float,
                        default=0.32, dest="theta")

    args = parser.parse_args()

    main(args.dbfile, args.misfile, args.supfile, args.statfile,
         args.filterfile, args.mis_filter, args.gamma, args.theta)
