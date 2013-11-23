# coding=utf-8
#-------------------------------------------------------------------------------
#License GNU/GPL v3
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#
#2013, Alexandre Manhaes Savio
#Use this at your own risk!
#-------------------------------------------------------------------------------

def save_fig_to_png(fig, fname, facecolor=None, dpi=300):
    """
    @param fig:
    @param fname:
    @param facecolor:
    @param dpi:
    @return:
    """

    import pylab as plt

    print ("Saving " + fname)
    fig.set_size_inches(22,16)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight', pad_inches=0,
                dpi=dpi, facecolor=facecolor)
    plt.close(fig)


def subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, yvals, yvariances, yrange,
                 c,  clor='k', mark='o', sty='_', show_legend=True):
    """
    @param plt:
    @param ax:
    @param xlabel:
    @param ylabel:
    @param prefs_thrs:
    @param yvals:
    @param yvariances:
    @param yrange:
    @param c:
    @param clor:
    @param mark:
    @param sty:
    @param show_legend:
    """
    line = ax.errorbar(prefs_thrs, yvals, yerr=yvariances, color=clor, marker=mark, ls=sty, label=c, lw=2, elinewidth=2, capsize=5)
    ax.set_xlabel(xlabel, size='xx-large')
    ax.set_ylabel(ylabel, labelpad=10, size='xx-large')
    #ax.set_yticks(yrange)
    plt.yticks(yrange, size='x-large')
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
    plt.xticks(prefs_thrs, rotation='vertical', size='x-large')
    if show_legend:
        ax.legend(loc=3)


def plot_results (results, wd, dataf, prefs_methods, prefs_thrs, clf_methods):
    """

    @param results:
    @param wd:
    @param masks:
    @param subjsf:
    @param prefs_methods:
    @param prefs_thrs:
    @param clf_methods:
    @return:
    """

    ##SHOW FINAL RESULTS
    import pylab as plt

    yrange = np.arange(0.5, 1.0, 0.1)

    #colors = ['ro-', 'gx-', 'bs-']
    colors = ['r', 'g', 'b', 'k', 'y', 'm']
    styles = ['-', '--', ':', '_']
    markrs = ['D', 'o', 'v', '+', 's', 'x']

    for fi, f in enumerate(subjsf):

        f = subjsf[fi]
        print f
        resf = results[f]
        print len(resf)

        for p in prefs_methods:
            print p
            resfp = aizc.filter_objlist (resf, 'prefs', p)

            fig = plt.figure(p + '_' + os.path.basename(f))

            i = 0
            for c in clf_methods:
                print c
                resfpc = aizc.filter_objlist (resfp, 'cl', c)

                clor = colors[i]
                mark = markrs[i]
                sty  = styles[i]
                i += 1

                # getting accuracy, spec, sens
                maccs, msens, mspec, mprec, mfone, mrauc = [], [], [], [], [], []
                vaccs, vsens, vspec, vprec, vfone, vrauc = [], [], [], [], [], []
                for t in prefs_thrs:
                    resfpct = aizc.filter_objlist (resfpc, 'prefs_thr', t)[0]

                    #metrics[i, :] = np.array([acc, sens, spec, prec, f1, roc_auc])
                    metrs = np.array(resfpct.metrics)

                    means = metrs.mean(axis=0)
                    varis = metrs.var (axis=0)

                    #get mean accuracy, sens and spec
                    maccs.append(means[0])
                    msens.append(means[1])
                    mspec.append(means[2])
                    mprec.append(means[3])
                    mfone.append(means[4])
                    mrauc.append(means[5])

                    #get var accuracy, sens and spec
                    vaccs.append(varis[0])
                    vsens.append(varis[1])
                    vspec.append(varis[2])
                    vprec.append(varis[3])
                    vfone.append(varis[4])
                    vrauc.append(varis[5])

                xlabel = 'threshold'

                ylabel = 'accuracy'
                ax     = plt.subplot(2,3,1)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, maccs, vaccs, yrange, c, clor, mark, sty, True)

                ylabel = 'sensitivity'
                ax     = plt.subplot(2,3,2)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, msens, vsens, yrange, c, clor, mark, sty, False)

                figure_title = str.upper(p[0]) + p[1:] + ' on ' + str(os.path.basename(f))
                plt.text(0.5, 1.08, figure_title, horizontalalignment='center', fontsize=20, transform = ax.transAxes, fontname='Ubuntu')
                #plt.title (figure_title)

                ylabel = 'specificity'
                ax     = plt.subplot(2,3,3)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mspec, vspec, yrange, c, clor, mark, sty, False)

                ylabel = 'precision'
                ax     = plt.subplot(2,3,4)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mprec, vprec, yrange, c, clor, mark, sty, False)

                ylabel = 'F1-score'
                ax     = plt.subplot(2,3,5)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mfone, vfone, yrange, c, clor, mark, sty, False)

                ylabel = 'ROC AUC'
                ax     = plt.subplot(2,3,6)
                subplot_this(plt, ax, xlabel, ylabel, prefs_thrs, mrauc, vrauc, yrange, c, clor, mark, sty, False)

            #fig.show()
            #raw_input("Press Enter to continue...")
            fname = p + '_' + os.path.basename(f) + '.png'
            fname = fname.lower()
            plot_fname = os.path.join(wd, fname)
            save_fig_to_png(fig, plot_fname, 'white')
            plt.close()