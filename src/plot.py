# You might need to `conda install seaborn` before running this script.
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
sns.set_theme()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", help="experiment version")
    args = parser.parse_args()

    assert args.v != None, "Must include experiment number!"

    if os.path.exists('./results/'):
        pass
    else:
        os.makedirs('./results/')

    results_file = open(f"./results/results-{args.v}.txt", "r")
    experiment = None
    x = []
    y = []
    for (i, line) in enumerate(results_file):
        if i == 0:
            continue
        nums = line.split()
        experiment = nums[0]
        x.append(int(nums[1]))
        y.append(int(nums[2]))

    plot = sns.lineplot(x=x, y=y)
    plot.set(xscale="log", xlabel="Top-k", ylabel="Accuracy",
             title=f"Experiment Version: {args.v}")
    plt.ylim(0, 100)

    # adding labels
    for xy in zip(x, y):
        plot.annotate('(%s, %s)' % xy, xy=xy, xytext=(10,-10), textcoords='offset points')

    fig = plot.get_figure()
    fig.savefig(f"./results/plot-{args.v}.png", bbox_inches="tight")

if __name__ == '__main__':
    main()