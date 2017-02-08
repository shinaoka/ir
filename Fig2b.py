import numpy as npimport matplotlib.pylab as pltimport pylabfrom kernel import KernelBasisDEfrom scipy.special import legendreparams = {    'backend': 'ps',    'axes.labelsize': 18,    'text.fontsize': 18,    'axes.titlesize': 18,    'legend.fontsize': 18,    'xtick.labelsize': 18,    'ytick.labelsize': 18,    'text.usetex': True,    }pylab.rcParams.update(params)def find_zeros(x, y):    r = []    for i in range(len(y)-1):        if y[i] * y[i+1] < 0:            zero = (x[i] * np.abs(y[i+1]) + x[i+1] * np.abs(y[i]))/(np.abs(y[i]) + np.abs(y[i+1]))            r.append(zero)    return np.array(r)results = []N = 1001il_list = [0, 1, 2]marker_list = ['o', 'x', '+', 'v']marker_list = ['+', 'v', 'o', 'v']color_list = ['b', 'r', 'g', 'k']wmax_list = [10.0, 50.0, 100.0]#wmax_list = [0.5, 1.0]ls_list = ['-', ':', '--']ikernel = 0for kernel in ['Fermionic', 'Bosonic']:    f, axes = plt.subplots(len(il_list), 2, figsize=(10, 10), sharex=True, sharey=False)    #f.subplots_adjust(wspace=0.25, hspace=0.2, right = 0.98, top=0.95)    f.subplots_adjust(wspace=0.25, hspace=0.1, right = 0.98, top=0.95)    axes = axes.transpose()    #axes[0,0].set_xlim([0.0,0.001])    iwmax = 0    for wmax in wmax_list:        kb = KernelBasisDE(N, N, wmax, 1e-12, kernel=kernel)        for il in il_list:            label_str = r'$\Lambda='+str(int(wmax))+'$'            if il != 1:                label_str = ''            axes[0,il].plot(kb.x_points(), kb.x_basis()[:,il]/kb.x_basis()[-1,il], label=label_str, marker='', color=color_list[iwmax], ls=ls_list[iwmax], lw=2)            axes[1,il].plot(kb.omega_points()/wmax, kb.omega_basis()[:,il]/kb.omega_basis()[-1,il], label=label_str, marker='', color=color_list[iwmax], ls=ls_list[iwmax], lw=2)            #axes[1,il].plot(kb.omega_points()/wmax, (kb.omega_points()/wmax)**(-1), label=label_str, marker='', color='g', ls=':', lw=0.5)            axes[0,il].set_xlim([-1,1])            axes[1,il].set_xlim([-1,1])            #axes[1,il].set_xscale("log")            #axes[1,il].set_yscale("log")        iwmax += 1    for il in range(len(il_list)-1):        plt.setp(axes[ikernel,il].get_xticklabels(), visible=False)    for il in range(len(il_list)):        #axes[0,il].set_title('$l='+str(il)+'$ ('+kernel[0]+')')        #axes[1,il].set_title('$l='+str(il)+'$ ('+kernel[0]+')')        lg = legendre(il)        x = np.linspace(-1,1,10000)        axes[0,il].plot(x, lg(x), label='', marker='', color='k', ls='-', lw=0.5)        axes[1,il].plot(x, lg(x), label='', marker='', color='k', ls='-', lw=0.5)        #axes[0,il].annotate(r"$l="+str(il)+'$', xy=(0.05, 0.9), fontsize=18, color='k', xycoords='axes fraction')        #axes[1,il].annotate(r"$l="+str(il)+'$', xy=(0.05, 0.9), fontsize=18, color='k', xycoords='axes fraction')        axes[0,il].set_title(r"$l="+str(il)+'$')        axes[1,il].set_title(r"$l="+str(il)+'$')        #plt.grid()    ikernel += 1    for il in range(len(il_list)):        axes[0,il].set_ylabel('$u_l(x)/u_l(1)$')        axes[1,il].set_ylabel('$v_l(y)/v_l(1)$')        #axes[0,il].plot([-1,1], [0,0], label='', marker='', color='k', ls='-', lw=0.5)        #axes[1,il].plot([-1,1], [0,0], label='', marker='', color='k', ls='-', lw=0.5)        axes[0,-1].set_xlabel('$x$')    axes[1,-1].set_xlabel('$y$')        axes[0,1].legend(loc='best',shadow=True,frameon=False)    axes[1,1].legend(loc='best',shadow=True,frameon=False)        plt.tight_layout()    plt.savefig("Fig2b"+kernel[0]+".pdf", transparent=True)