#Include depedencies
import numpy as np
from numpy import dot, pi, exp, sqrt, inf
from numpy.linalg import norm
import matplotlib.pylab as plt
from scipy.stats import linregress
from matplotlib import cm
from ritsar import signal as sig
from ritsar import phsTools
from scipy.interpolate import interp1d
import cupy as cp

import multiprocessing as mp

def backprojection_gpu(phs, platform, img_plane, taylor = 20, upsample = 7, pbypsave=0):
##############################################################################
#                                                                            #
#  This is the Backprojection algorithm.  The phase history data as well as  #
#  platform and image plane dictionaries are taken as inputs.  The (x,y,z)   #
#  locations of each pixel are required, as well as the size of the final    #
#  image (interpreted as [size(v) x size(u)]).                               #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    nsamples    =   platform['nsamples']
    npulses     =   platform['npulses']
    k_r         =   platform['k_r']
    pos         =   platform['pos']
    delta_r     =   platform['delta_r']
    u           =   img_plane['u']
    v           =   img_plane['v']
    r           =   img_plane['pixel_locs']

    # Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[nsamples//2]

    # Create window
    win_x = sig.taylor(nsamples, taylor)
    win_x = np.tile(win_x, [npulses, 1])

    win_y = sig.taylor(npulses, taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1, nsamples])

    win = win_x * win_y

    # Filter phase history
    filt = np.abs(k_r)
    phs_filt = phs * filt * win

    N_fft = 2 ** (int(np.log2(nsamples * upsample)) + 1)
    phs_pad = sig.pad(phs_filt, [npulses, N_fft])  ## 필터링 된 행렬을 N_fft 값에 맞추어 padding 한 행렬, 이 행렬에 에너지에 대한 정보가 있음.

    # Filter phase history and perform FT w.r.t t
    # Q = sig.ft(phs_pad)
    # Q_mag = np.abs(Q[:,:N_fft//2]) + np.abs(Q[:, -(N_fft//2):])
    Q = sig.ft(phs_pad)
    dr = np.linspace(-nsamples * delta_r / 2, nsamples * delta_r / 2, N_fft)
    # reverse_dr = np.linspace(nsamples * delta_r / 2, -nsamples * delta_r / 2, N_fft)

    ## Save PLOT
    # for i in range (npulses):
    #     plt.plot(dr,np.abs(Q[i]))
    #     plt.title(f"Q[{i}]")
    #     plt.savefig(f"PLOT/Q[{i}]")
    #     plt.close()

    mempool = cp.get_default_memory_pool()

    # Perform backprojection for each pulse

    img_gpu = kernel2(Q, pos, r, dr, nu, nv, npulses, k_c, pbypsave)
    ##### single_img_set is in cpu

    mempool.free_all_blocks()


    return img_gpu
@cp.fuse()
def kernel2(Q, pos, r, dr, nu, nv, npulses, k_c, pbypsave):
    img_gpu = cp.zeros(nu * nv) + 0j
    Q_gpu = cp.array(Q)
    pos_gpu = cp.array(pos)
    target_loc_gpu = cp.array(r)
    dr_gpu = cp.array(dr)
    k_c_gpu = cp.array(k_c)

    print("Start BP")
    for i in range(npulses):
        pos_tiled = cp.tile(pos_gpu[i][0:3], (nu * nv, 1)).T
        dr_i = cp.linalg.norm(target_loc_gpu - pos_tiled, axis=0)

        Q_real = cp.interp(dr_i, dr_gpu, Q_gpu[i].real, right=1)
        Q_imag = cp.interp(dr_i, dr_gpu, Q_gpu[i].imag, right=0)

        Q_hat = cp.add(Q_real, cp.multiply(1j, Q_imag))

        img_gpu += cp.multiply(Q_hat, cp.exp(-1j * k_c_gpu * dr_i))
    return img_gpu

def backprojection(phs, platform, img_plane, taylor=20, upsample=6, prnt=True):
    ##############################################################################
    #                                                                            #
    #  This is the Backprojection algorithm.  The phase history data as well as  #
    #  platform and image plane dictionaries are taken as inputs.  The (x,y,z)   #
    #  locations of each pixel are required, as well as the size of the final    #
    #  image (interpreted as [size(v) x size(u)]).                               #
    #                                                                            #
    ##############################################################################

    # Retrieve relevent parameters
    nsamples = platform['nsamples']
    npulses = platform['npulses']
    k_r = platform['k_r']
    pos = platform['pos']
    delta_r = platform['delta_r']
    u = img_plane['u']
    v = img_plane['v']
    r = img_plane['pixel_locs']

    # Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[nsamples // 2]  # centered frequency

    # Create window
    win_x = sig.taylor(nsamples, taylor)
    win_x = np.tile(win_x, [npulses, 1])

    win_y = sig.taylor(npulses, taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1, nsamples])

    win = win_x * win_y

    # Filter phase history
    filt = np.abs(k_r)
    phs_filt = phs * filt * win  ## 적절한 값의 필터를 사용하여 재구성된 배열

    # Zero pad phase history
    N_fft = 2 ** (int(np.log2(nsamples * upsample)) + 1)
    phs_pad = sig.pad(phs_filt,[npulses, N_fft])  ## 필터링 된 행렬을 N_fft 값에 맞추어 padding 한 행렬, 이 행렬에 에너지에 대한 정보가 있음.

    # Filter phase history and perform FT w.r.t t
    # Q = sig.ft(phs_pad)
    # dr = np.linspace(-nsamples * delta_r / 2, nsamples * delta_r / 2, N_fft)
    Q = np.fft.fft(phs_pad)
    dr = np.linspace(0, nsamples * delta_r , N_fft)

    cut = 0
    if cut:
        Q_cut = np.zeros_like(Q)
        Q_cut[:,:N_fft-200] = Q[:,:N_fft-200]
        Q = Q_cut

    front = 0
    back = 0
    if front:
        Q_cut = Q[:, front:-back]
        Q_after = np.zeros_like(Q)
        for i in range(npulses):
            Q_after[i] = np.interp(np.linspace(0, 1, N_fft), np.linspace(0, 1, N_fft-(front+back)), Q_cut[i])
        Q = Q_after



    # Perform backprojection for each pulse
    img = np.zeros(nu * nv) + 0j  ## samples * pulse 개수의 배열 생성
    for i in range(npulses):  ## 펄스의 개수만큼 반복문
        if prnt:
            print("Calculating backprojection for pulse %i" % i)
        r0 = np.array([pos[i]]).T  ## platform에서 생성된 pos 행렬의 i번째 행을 가져옴. 이때 tranpose를 취하여 i번째 행 인덱스의 정보를 배열로 나열함.
        dr_i = norm(r - r0, axis=0)  ## r0를 norm을 취하면 원점에서 펄스까지의 거리가 나옴. 이 값에서 펄스-픽셀 거리를 계산한 값을 빼서 거리 값을 구한다?

        Q_real = np.interp(dr_i, dr, Q[i].real)
        Q_imag = np.interp(dr_i, dr, Q[i].imag)
        # Q_real = np.interp(dr_i, dr, Q[i].real, right=1)
        # Q_imag = np.interp(dr_i, dr, Q[i].imag, right=0)
        Q_hat = Q_real + 1j * Q_imag
        img += Q_hat * np.exp(-1j * k_c * dr_i)

    # r0 = np.array([pos[npulses//2]]).T
    # dr_i = norm(r0)-norm(r-r0, axis = 0)
    # img = img * np.exp(1j * k_c * dr_i)
    img = np.reshape(img, [nv, nu])[::-1, :]
    return (img)

def matched_filter(phs,platform):
    Tx = platform['Tx']
    t = platform['t']
    matched_filter = np.conj(Tx[::-1])
    Rx = phs
    filtered_signal = np.convolve(Rx,matched_filter,mode='same')

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Original and Received Signal")
    plt.plot(t, np.abs(Tx), label='Original Signal')
    plt.plot(t, np.abs(Rx), label='Received Signal with Noise')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Output of the Matched Filter")
    plt.plot(t, np.abs(filtered_signal))
    plt.tight_layout()
    plt.show()
    a=1

def RGC(phs, platform,test_name):
    nsamples = platform['nsamples']
    npulses = platform['npulses']
    Q = sig.ft(phs)
    Q_T = np.transpose(Q)
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(Q_T[256:,:]), aspect='auto', cmap='gray',origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title(f'{test_name}')
    plt.xlabel('Pulses')
    plt.ylabel('Range Bins')
    plt.tight_layout()
    plt.show()

def img_plane_dict_0326(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']

    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    """ 여기만 확인"""
    u_start = 0
    u_end = 10

    v_start = 0
    v_end = 17
    res = 1024
    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0331(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']

    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    u_start = 4.85
    u_end = 13.55

    v_start = 4.92
    v_end = 30.92
    res = 1024
    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0401_D(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']

    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    u_start = 0
    u_end = 90.505

    v_start = 0
    v_end = 43.17

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0401_C(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']
    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    u_start = 0
    u_end = 55

    v_start = 0
    v_end = 43.17

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0409_C(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']
    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    u_start = 0
    u_end = 37

    v_start = 0
    v_end = 86.34

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0528_linear(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']
    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    u_start = 0
    u_end = 18

    v_start = 0
    v_end = 20

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0528_circular(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']
    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    u_start = 20
    u_end = 50

    v_start = 0
    v_end = 25

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0607(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']
    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    u_start = -40
    u_end = 40

    v_start = -40
    v_end = 40

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0610(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']
    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    delta_r = platform['delta_r']
    u_start = -5
    u_end = 20
    res_u = round((u_end-u_start)/delta_r)
    v_start = 0
    v_end = 10
    res_v = round((v_end - v_start) / delta_r)

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'res_u': res_u,
            'res_v': res_v,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)

def img_plane_dict_0624(platform, res_factor=1.0, n_hat = np.array([0 ,0 ,1]), aspect = 1, upsample = True):
    ##############################################################################
    #                                                                            #
    #  This function defines the image plane parameters.  The user specifies the #
    #  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
    #  image plane whose pixels are sized at the theoretical resolution limit    #
    #  of the system (derived using delta_r which in turn was derived using the  #
    #  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
    #  defaults to nsamples/npulses.                                             #
    #                                                                            #
    #  'n_hat' is a user specified value that defines the image plane            #
    #  orientation w.r.t. to the nominal ground plane.                           #
    #                                                                            #
    ##############################################################################

    nsamples = platform['nsamples']
    npulses = platform['npulses']
    # Import relevant platform parameters
    R_c = platform['R_c'] # 중간 펄스 위치

    # Define image plane parameters
    if upsample:
        nu= 2** int(np.log2(nsamples) + bool(np.mod(np.log2(nsamples), 1)))
        nv = 2 ** int(np.log2(npulses) + bool(np.mod(np.log2(npulses), 1)))
    else:
        nu = nsamples
        nv = npulses

    # Define resolution.  This should be less than the system resolution limits
    du = platform['delta_r'] * res_factor * nsamples / nu  # range resolution
    dv = aspect * du  # range resolution

    delta_r = platform['delta_r']
    u_start = 25
    u_end = 45
    res_u = 340
    v_start = 5
    v_end = 25
    res_v = 170

    res = 1024

    u = np.linspace(u_start, u_end, res)
    v = np.linspace(v_start, v_end, res)

    # Derive image plane spatial frequencies
    k_u = 2 * pi * np.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)  # sample들의 공간주파수
    k_v = 2 * pi * np.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)  # pulse들의 공간주파수

    # Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c) / norm(np.cross(n_hat, R_c))  # plane의 x축 y축 같은 의미인가...?
    u_hat = np.cross(v_hat, n_hat) / norm(np.cross(v_hat, n_hat))

    # Represent u and v in (x,y,z)
    [uu, vv] = np.meshgrid(u, v)
    uu = uu.flatten()
    vv = vv.flatten()
    zz = np.zeros_like(vv)

    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T
    )))
    b = np.asmatrix(np.vstack((uu, vv, zz)))
    # pixel_locs = np.asarray(A*b)
    pixel_locs = np.asarray(b)  ## 픽셀들의 x,y,z값을 행렬로 나타냄. 각 픽셀당 x,y,z값을 가지므로 256 * 512 = 131072개의 행이 존재

    # Construct dictionary and return to caller
    img_plane = \
        {
            'n_hat': n_hat,
            'u_hat': u_hat,
            'v_hat': v_hat,
            'du': du,
            'dv': dv,
            'u_start': u_start,
            'u_end': u_end,
            'v_start': v_start,
            'v_end': v_end,
            'res': res,
            'res_u': res_u,
            'res_v': res_v,
            'u': u,
            'v': v,
            'k_u': k_u,
            'k_v': k_v,
            'pixel_locs': pixel_locs  # 3 x N_pixel array specifying x,y,z location
            # of each pixel
        }

    return (img_plane)