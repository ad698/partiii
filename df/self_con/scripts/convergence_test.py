# coding: utf-8

from actest_py import *
import numpy as np
import matplotlib.pyplot as plt

G = GalPot("../../Torus/pot/PJM11.Tpot")
eps = 1e-9
O = Orbit(G, eps)
alpha = -12.
A = Actions_AxisymmetricStackel_Fudge(G, alpha)
eta, MaxIterations, DeltaJ = 1e-8, 5, 1e-4
T = IterativeTorusMachine(A, G, eta, MaxIterations, DeltaJ)

def integrate_orbit(X, O, t, steps):
    P = X
    results = X
    for i in np.arange(steps):
        P = O.integrate(P, t)
        results = np.vstack((results, P))
    return results


def calc():
    for i in np.arange(4., 12.1, 2.):
        vmin = np.sqrt(-G.forces(np.array([i, 0, 0]))[0] * i)
        for j in np.arange(0.5 * vmin,
                           # np.sqrt(2. * (
                           #     G(np.array([10. * i, 0., 10. * i])) -
                           #     G(np.array([i, 0, 0]))))
                           .81 * vmin, vmin * 0.1):
            for k in np.arange(0.2, np.pi / 2., np.pi / 8.-0.201/4.):
                # i, j, k = 8.0, 107.738447581, 0.835398163397
                # vmin = np.sqrt(-G.forces(np.array([i, 0, 0]))[0] * i)
                Xlaunch = np.array([i, 0.01, 0.01,
                                    j * np.cos(k), vmin, j * np.sin(k)])
                resL = integrate_orbit(Xlaunch, O, 0.0005, 5000)
                Fudg = np.array(map(lambda i: A.actions(i,True),
                                    resL[::len(resL) / 10]))
                # plt.plot(Fudg.T[2])
                # plt.savefig('tmp.eps')
                Iter = np.array(map(lambda i: T.actions_and_freqs(i),
                                   resL[::len(resL) / 10]))
                # print Iter
                print i, j, k, np.mean(Fudg.T[0]), np.mean(Fudg.T[2]), np.std(Fudg.T[0],ddof=1), np.std(Fudg.T[2],ddof=1), np.mean(Fudg.T[3]), np.mean(Fudg.T[5]), np.std(Fudg.T[3],ddof=1), np.std(Fudg.T[5],ddof=1), np.mean(Iter.T[0]), np.mean(Iter.T[2]), np.std(Iter.T[0],ddof=1), np.std(Iter.T[2],ddof=1), np.mean(Iter.T[3]), np.mean(Iter.T[5]), np.std(Iter.T[3],ddof=1), np.std(Iter.T[5],ddof=1)


def plot():
    count = 0
    f, a = plt.subplots(1, 2)
    for i in np.arange(2., 9.9, 2.):
        vmin = np.sqrt(-G.forces(np.array([i, 0, 0]))[0] * i)
        for j in np.arange(0.01 * vmin,
                           # np.sqrt(2. * (
                           #     G(np.array([10. * i, 0., 10. * i])) -
                           #     G(np.array([i, 0, 0]))))
                           .5 * vmin, vmin * 0.15):
            for k in np.arange(0.05, np.pi / 2. - 0.4, np.pi / 16.):
                # i, j, k = 6.0, 70.5050162624, 0.442699081699
                # i, j, k = 8.0, 107.738447581, 0.835398163397
                # vmin = np.sqrt(-G.forces(np.array([i, 0, 0]))[0] * i)
                Xlaunch = np.array([i, 0.01, 0.01,
                                    j * np.cos(k), vmin, j * np.sin(k)])
                resL = integrate_orbit(Xlaunch, O, 0.0005, 5000)
                if(count == 62 or count == 94):
                    a[0].plot(np.sqrt(resL.T[0] ** 2 + resL.T[1] ** 2), resL.T[2],
                              lw=0.5, alpha=0.5)
                count += 1
    # Fudg = np.genfromtxt("converg_results.txt")
    # a[1].plot(np.log(Fudg.T[0]), np.log(Fudg.T[2]), '.')
    f.savefig('converg.png', bbox_inches='tight')


def fancy():
    g = np.genfromtxt("converg_results.dat")
    f, a = plt.subplots(2, 1, figsize=[3.32, 5.])
    plt.subplots_adjust(hspace=0.3)
    h = np.logspace(-7., 1.)
    a[0].set_xscale('log')
    a[0].set_yscale('log')
    a[0].set_xlabel(r'$\log(\Delta J_{R,{\rm fudge}}/{\rm kpc\,km\,s}^{-1})$')
    a[0].set_ylabel(r'$\log(\Delta J_{R,{\rm iter}}/{\rm kpc\,km\,s}^{-1})$')
    a[0].plot(h, h, ':')
    a[0].plot(g.T[5], g.T[9], 'k.')
    # a[0].plot(g.T[5],1e-5*g.T[3],'r.',alpha=0.2)
    a[1].set_xscale('log')
    a[1].set_yscale('log')
    a[1].set_xlabel(r'$\log(\Delta J_{z,{\rm fudge}}/{\rm kpc\,km\,s}^{-1})$')
    a[1].set_ylabel(r'$\log(\Delta J_{z,{\rm iter}}/{\rm kpc\,km\,s}^{-1})$')
    a[1].plot(g.T[6], g.T[10], 'k.')
    a[1].plot(h, h, ':')
    # a[1].plot(g.T[6],1e-5*g.T[4],'r.',alpha=0.2)
    f.savefig('converg_acts.pdf', bbox_inches='tight')


def fancy2():
    g = np.genfromtxt("converg_test.results")
    f,ax = plt.subplots(2,1,figsize=[3.32, 5.])
    plt.subplots_adjust(hspace=0.4)
    # ax[0].set_xscale('log')
    # ax[0].set_yscale('log')
    # ax[0].set_ylim(0.001,1000.)
    ax[0].plot(g.T[11],g.T[12],'k.')
    ax[0].set_xlabel(r'$J_R/{\rm kpc\,km\,s}^{-1}$')
    ax[0].set_ylabel(r'$J_z/{\rm kpc\,km\,s}^{-1}$')
    h = np.logspace(-2., 2.)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\Delta J_{\rm fudge}/{\rm kpc\,km\,s}^{-1}$')
    ax[1].set_ylabel(r'$\Delta J_{\rm iter}/{\rm kpc\,km\,s}^{-1}$')
    ax[1].plot(h, h, ':')
    ax[1].plot(g.T[5], g.T[13], 'k.',markersize=5, label=r'$J_R$')
    # ax[1].plot(g.T[16], 1e-4*np.max(g.T[3:5].T,axis=1), 'b.',markersize=5, label=r'$J_R$')
    ax[1].plot(g.T[6], g.T[14], 'rx', markersize=3, label=r'$J_z$')
    l = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),ncol = 2)
    l.draw_frame(False)
    f.savefig('converg_acts2.pdf', bbox_inches='tight')


# calc()
# plot()
fancy2()
