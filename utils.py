import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2

def rotate_points(pts : np.ndarray, theta : float, centre=(0.0, 0.0)):
    """
    Rotate 2-D points anticlockwise by *theta* (radians) about *centre*.

    pts    : (N, 2) array of [x, y] rows
    theta  : rotation angle in radians (positive = CCW)
    centre : tuple (cx, cy) – default (0, 0)

    Returns the rotated (N, 2) array.
    """
    pts = np.asarray(pts, dtype=float)
    cx, cy = centre
    c, s   = np.cos(theta), np.sin(theta)

    # shift → rotate → shift back
    x, y   = pts.T
    x_rel, y_rel = x - cx, y - cy
    x_rot  =  c * x_rel - s * y_rel + cx
    y_rot  =  s * x_rel + c * y_rel + cy
    return np.column_stack([x_rot, y_rot])

def resample(coords : np.ndarray, values : np.ndarray, x_disp : float, y_disp : float, theta_rotation : float, plot=True):
    """
    Resampe and linearly transform FEA results. 
    Rotate and linearly displace 2-D points at *coords* postions with field value of *values* anticlockwise by *theta_rotation* (radians) and *x_disp*/*y_disp* (meters).

    **Parameters**

    coords  :   (N, 2) array of [x, y] rows
    values  :   (N, 1) array of z-axis field valutes
    theta_rotation   :   rotation angle in radians (positive = CCW)
    x_disp  :   x displacement in meters
    y_disp  :   y displacement in meters

    Returns the resampled (5, 5) array.
    """
    coords = rotate_points(coords,theta_rotation)
    nx, ny = 8, 8
    xi  = np.linspace(-0.01+x_disp, 0.01+x_disp, nx)
    yi  = np.linspace(-0.01+y_disp, 0.01+y_disp, ny)
    X, Y = np.meshgrid(xi, yi)

    V = griddata(coords[:,:], values, (X, Y), method='linear')

    if plot:
        plt.figure(num=None)
        plt.imshow(V)
        plt.colorbar(label="field value")
        plt.ylim(-0.5,7.5)
        plt.xlim(-0.5,7.5)
        plt.show()

    return(V)

def get_ticks(coords):
    x_tick_const = (-coords[:,0].min()*0.42 + coords[:,0].max()*0.42)/5
    y_tick_const = (-coords[:,1].min()*0.39 + coords[:,1].max()*0.39)/5
    return [x_tick_const,y_tick_const]

def get_mutational(start=-0.2, end=0.6):
    return np.random.uniform(start, end)

def get_scalable(start=1, end=1.3):
    return np.random.uniform(start, end)

def get_rotational(start=0, end=2):
    return np.pi*np.random.uniform(start, end)

def get_ran_pos(sqr_const=1.54):
    square = np.array([[-sqr_const,-sqr_const],
                  [sqr_const,-sqr_const],
                  [sqr_const,sqr_const],
                  [-sqr_const,sqr_const]]).astype('float32')

    ns = square*get_scalable()
    ns = rotate_points(square,get_rotational())
    for i in range(4):
        ns[i, 0] += get_mutational()  # mutate x
        ns[i, 1] += get_mutational()  # mutate y
    return square, ns

def get_combined(coords,values,test):
    _, ns = get_ran_pos()

    [Xt, Yt] = get_ticks(coords)

    a = resample(coords, values, ns[0,0]*Xt, ns[0,1]*Yt, get_rotational(), test)
    b = resample(coords, values, ns[1,0]*Xt, ns[1,1]*Yt, get_rotational(), test)
    c = resample(coords, values, ns[2,0]*Xt, ns[2,1]*Yt, get_rotational(), test)
    d = resample(coords, values, ns[3,0]*Xt, ns[3,1]*Yt, get_rotational(), test)

    return np.stack([a, b, c, d],axis=0), ns

x_coord_layer = np.array([[-3,-2,-1,1,2,3],
                          [-3,-2,-1,1,2,3],
                          [-3,-2,-1,1,2,3],
                          [-3,-2,-1,1,2,3],
                          [-3,-2,-1,1,2,3],
                          [-3,-2,-1,1,2,3]])/3

y_coord_layer = x_coord_layer.T

def get_time_evolution(coords,values,speed1=0.5,speed2=-0.5,speed3=-0.5, speed4=0.5,starting_offset =6, test=False):
    [Xt, Yt] = [0.0005,0.0005]
    times = np.arange(10)
    combined = []

    for i in range(1):
        for time in times:
            a = resample(coords, values, (time*speed1+starting_offset+8)*Xt, (starting_offset-8)*Yt, 0, test)
            b = resample(coords, values, (0-starting_offset+8)*Xt, (time*(speed2)-starting_offset-8)*Yt, 0, test)
            c = resample(coords, values, (starting_offset-8)*Xt, (time*(speed4)+starting_offset+8)*Yt, 0, test)
            d = resample(coords, values, (time*speed3-starting_offset-8)*Xt, (0-starting_offset+8)*Yt, 0, test)
            combined.append(a+b+c+d)
            
        for t in times:
            time = 10-t
            a = resample(coords, values, (time*speed1+starting_offset+8)*Xt, (starting_offset-8)*Yt, 0, test)
            b = resample(coords, values, (0-starting_offset+8)*Xt, (time*(speed2)-starting_offset-8)*Yt, 0, test)
            c = resample(coords, values, (starting_offset-8)*Xt, (time*(speed4)+starting_offset+8)*Yt, 0, test)
            d = resample(coords, values, (time*speed3-starting_offset-8)*Xt, (0-starting_offset+8)*Yt, 0, test)
            combined.append(a+b+c+d)

    for i in range(1):
        for time in times:
            a = resample(coords, values, (time*speed1+starting_offset+7.6)*Xt, (starting_offset-7.6)*Yt, 0, test)
            b = resample(coords, values, (0-starting_offset+7.6)*Xt, (time*(speed2)-starting_offset-7.6)*Yt, 0, test)
            c = resample(coords, values, (starting_offset-7.6)*Xt, (time*(speed4)+starting_offset+7.6)*Yt, 0, test)
            d = resample(coords, values, (time*speed3-starting_offset-7.6)*Xt, (0-starting_offset+7.6)*Yt, 0, test)
            combined.append(a+b+c+d)
            
        for t in times:
            time = 10-t
            a = resample(coords, values, (time*speed1+starting_offset+7.6)*Xt, (starting_offset-7.6)*Yt, 0, test)
            b = resample(coords, values, (0-starting_offset+7.6)*Xt, (time*(speed2)-starting_offset-7.6)*Yt, 0, test)
            c = resample(coords, values, (starting_offset-7.6)*Xt, (time*(speed4)+starting_offset+7.6)*Yt, 0, test)
            d = resample(coords, values, (time*speed3-starting_offset-7.6)*Xt, (0-starting_offset+7.6)*Yt, 0, test)
            combined.append(a+b+c+d)
        

    return np.stack(combined,axis=0)

def gaussian_2d(h, w, cx, cy, sigma):
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return g

def combined_gaussians(params, U_obs, sigma=3):
    cx1, cy1, cx2, cy2, cx3, cy3, cx4, cy4, alpha1, alpha2, alpha3, alpha4, sigma1, sigma2, sigma3, sigma4 = params
    g1 = gaussian_2d(*U_obs.shape, cx1, cy1, sigma1)
    g2 = gaussian_2d(*U_obs.shape, cx2, cy2, sigma2)
    g3 = gaussian_2d(*U_obs.shape, cx3, cy3, sigma3)
    g4 = gaussian_2d(*U_obs.shape, cx4, cy4, sigma4)
    U_fit = alpha1 * g1 + alpha2 * g2 + alpha3 * g3 + alpha4 * g4
    loss = np.linalg.norm(U_fit - U_obs)**2
    return loss

def argmax_2d(image):
    return [np.argmax(image)//image.shape[0], np.argmax(image)%image.shape[0]]

def make_gaussian_field(h, w, center, sigma=0.25):
    """Returns a flattened 2D Gaussian centered at 'center'."""
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    X, Y = np.meshgrid(x, y)
    cy, cx = center
    g = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return g.flatten()  # shape: (h*w,)

def optimization_routine_3d(image, test=False):
    upscaled = cv2.resize(image, (20, 20), interpolation=cv2.INTER_CUBIC)
    upscaled /= np.max(upscaled)
    U=upscaled
    if test:
        plt.figure()
        plt.imshow(U)

    pt1 = np.array(argmax_2d(upscaled))
    inhib_factor = 1 - make_gaussian_field(20, 20, pt1, sigma=5).reshape([20,20])
    nim = inhib_factor*upscaled
    if test:
        plt.figure()
        plt.imshow(nim)
    pt2 = np.array(argmax_2d(nim))
    inhib_factor = 1 - make_gaussian_field(20, 20, pt2, sigma=5).reshape([20,20])
    nim = inhib_factor*nim
    if test:
        plt.figure()
        plt.imshow(nim)
    pt3 = np.array(argmax_2d(nim))
    inhib_factor = 1 - make_gaussian_field(20, 20, pt3, sigma=5).reshape([20,20])
    nim = inhib_factor*nim
    if test:
        plt.figure()
        plt.imshow(nim)
    pt4 = np.array(argmax_2d(nim))

    if test:
        print(pt1)
        print(pt2)
        print(pt3)
        print(pt4)
    
    from scipy.optimize import minimize

    H, W = U.shape
    initial_guess = [pt1[0], pt1[1], pt2[0], pt2[1], 
                     pt3[0], pt3[1], pt4[0], pt4[1], 
                     1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25]

    bounds = [
        (0, W-0.1), (0, H-0.1),   # x1, y1
        (0, W-0.1), (0, H-0.1),   # x2, y2
        (0, W-0.1), (0, H-0.1),   # x3, y3
        (0, W-0.1), (0, H-0.1),   # x4, y4
        (0, None), (0, None),  # alpha1, alpha2
        (0, None), (0, None),  # alpha3, alpha4
        (0, None), (0, None),  # sigma1, sigma4
        (0, None), (0, None),  # sigma3, sigma4
    ]

    res = minimize(combined_gaussians, initial_guess, args=(U,), bounds=bounds, method='L-BFGS-B')

    best_fit = res.x[8] * gaussian_2d(H, W, res.x[0], res.x[1], 3) + \
           res.x[9] * gaussian_2d(H, W, res.x[2], res.x[3], 3) + \
           res.x[10] * gaussian_2d(H, W, res.x[4], res.x[5], 3) + \
           res.x[11] * gaussian_2d(H, W, res.x[6], res.x[7], 3)
    
    if test:
        plt.figure()
        plt.imshow(image)
        
        from mpl_toolkits.mplot3d import Axes3D

        Z = best_fit

        x = np.arange(Z.shape[1])
        y = np.arange(Z.shape[0])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("3D Surface Plot")
        plt.tight_layout()
        plt.show()

    return res


from scipy.ndimage import generic_filter

def nanmean_filter(values):
    """Filter function that computes mean of non-NaN values."""
    vals = values[~np.isnan(values)]
    return np.mean(vals) if len(vals) > 0 else np.nan

def replace_nan_with_neighbors(array):
    mask = np.isnan(array)
    # Use a 3x3 window (including diagonals) for local averaging
    filled = generic_filter(array, nanmean_filter, size=3, mode='constant', cval=np.nan)
    # Replace NaNs with the computed local average
    array[mask] = filled[mask]
    return array

from concurrent.futures import ThreadPoolExecutor

def batch_prediction(combined: np.ndarray, workers: int = 1):
    n_frames = combined.shape[0]

    pos_array = np.empty((4, n_frames, 2), dtype=float)
    alpha_array = np.empty((4, n_frames), dtype=float)

    def process_frame(i):
        res = optimization_routine_3d(combined[i, :, :])

        positions = np.array([
            (res.x[0], res.x[1]),
            (res.x[2], res.x[3]),
            (res.x[4], res.x[5]),
            (res.x[6], res.x[7])
        ])
        alphas = np.array(res.x[8:12])
        return positions, alphas

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (pos, alpha) in enumerate(ex.map(process_frame, range(n_frames))):
                pos_array[:, i, :] = pos
                alpha_array[:, i] = alpha
    else:
        for i in range(n_frames):
            pos, alpha = process_frame(i)
            pos_array[:, i, :] = pos
            alpha_array[:, i] = alpha

    heights = np.sqrt(1.0 / alpha_array)

    predicted = np.concatenate((pos_array, heights[:, :, None]), axis=-1)
    return predicted


def get_volume(predicted_data: np.ndarray):
    p0 = predicted_data[0, :, :2]
    p1 = predicted_data[1, :, :2]
    p2 = predicted_data[2, :, :2]
    p3 = predicted_data[3, :, :2]

    all_points = np.stack([p0, p1, p2, p3], axis=0)

    max_x_spacing = np.max(all_points[:, :, 0], axis=0) - np.min(all_points[:, :, 0], axis=0)
    max_y_spacing = np.max(all_points[:, :, 1], axis=0) - np.min(all_points[:, :, 1], axis=0)

    dia = 0.5 * (max_x_spacing + max_y_spacing)

    return (dia / np.min(dia)) ** 3 - 1
