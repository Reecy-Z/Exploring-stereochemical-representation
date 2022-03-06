import numpy as np
import os
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

def plot_ANN_80_20_MAE_update(test_pred,
                              train_pred,
                              r2_test,
                              mae_test,
                              rmse_test,
                              train_target,
                              test_target,
                              epoch,
                              pos=111,
                              color = 6):

    fig = plt.figure(figsize=(10,10))
    # create subplot
    plt.subplot(pos)
    plt.grid(alpha=0.2)
    # plt.title(title, fontsize=15)
    colors=list(mcolors.TABLEAU_COLORS.keys())
    # add score patches
    r2_test_patch = mpatches.Patch(label="R$^2$_test = {:04.2f}".format(r2_test),color=mcolors.TABLEAU_COLORS[colors[color]])
    mae_test_patch = mpatches.Patch(label="MAE_test = {:04.3f}".format(mae_test),color=mcolors.TABLEAU_COLORS[colors[color]])
    rmse_test_patch = mpatches.Patch(label="RMSE_test = {:04.3f}".format(rmse_test),color=mcolors.TABLEAU_COLORS[colors[color]])
    plt.xlim(-1,3.5)
    plt.ylim(-1,3.5)
    plt.scatter(test_target, test_pred, alpha=0.2,color=mcolors.TABLEAU_COLORS[colors[color]],label = 'Test set')
    plt.scatter(train_target, train_pred, alpha=0.2,color=mcolors.TABLEAU_COLORS[colors[2]],label = 'Train set')
    plt.legend()
    plt.legend(handles=[r2_test_patch, mae_test_patch, rmse_test_patch], fontsize=12,loc='upper left')
    plt.plot(np.arange(-0.5,3.5,0.5), np.arange(-0.5,3.5,0.5), ls="--", c=".3")
    fig.text(0.5, 0.05, 'Observed ΔΔG/kcal/mol',ha='center', va='center', fontsize=15)
    fig.text(0.05, 0.5, 'Predicted ΔΔG/kcal/mol', ha='center', va='center', rotation='vertical', fontsize=15)
    for f in os.listdir('./'):
        if 'best_epoch' in f:
            os.remove(f)
    plt.savefig('best_epoch_' + str(epoch) +'.png', dpi = 300)
    # plt.show()

