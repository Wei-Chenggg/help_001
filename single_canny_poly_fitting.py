"""Single-line Canny + original Poly_Fitting experiment.
Run: python single_canny_poly_fitting.py
Dependencies: numpy, scipy, opencv-python, matplotlib
"""
import os
import pickle
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt


def create_folder(path):
    path = os.fspath(path)
    folder = os.path.dirname(path) if os.path.splitext(os.path.basename(path))[1] else path
    if folder and folder != '.':
        os.makedirs(folder, exist_ok=True)


def save_pkl(file_path, data):
    if not isinstance(file_path, str): file_path, data = data, file_path
    if not file_path.lower().endswith('.pkl'): file_path += '.pkl'
    create_folder(file_path)
    with open(file_path, 'wb') as f: pickle.dump(data, f)


def load_pkl(file_path):
    file_path = file_path.replace('.pkl', '')
    with open(file_path + '.pkl', 'rb') as f: return pickle.load(f)


# Original user class below; unmodified.
class Poly_Fitting():
    def __init__(self, print_detial=False):
        self.w = None
        self.print_detial = print_detial
        self.scaler = {}
        self.info = {}
    def upsample(self, x, y, ratio=2):
        xnew = np.linspace(x.min(), x.max(), int(len(x)*ratio))
        f = scipy.interpolate.interp1d(x, y)
        ynew = f(xnew)
        return xnew, ynew
    def _scale_fit(self, data, name, eps=1e-4):
        dmin = data.min(axis=0); dmax = data.max(axis=0)
        self.scaler[name] = {'min':dmin, 'max':dmax, 'eps':eps}
        return (data - dmin) / (dmax - dmin + eps)
    def _scale_apply(self, data, name):
        p = self.scaler[name]
        return (data - p['min']) / (p['max'] - p['min'] + p['eps'])
    def _scale_backward(self, data, name):
        p = self.scaler[name]
        return data * (p['max'] - p['min'] + p['eps']) + p['min']
    def _apply_drop_outlier(self, x, y, yfit, rate):
        e = abs(yfit - y)
        rank = np.argsort(e)
        num_samples = x.shape[0]
        preserve_samples = int(round(num_samples * (1-rate), 0))
        rank = rank[:preserve_samples]
        x = x[rank]; y = y[rank]; yfit = yfit[rank]
        return x, y, yfit
    def _least_square(self, x, y):
        A = self.convert2basis(x, order=self.order)
        if A is None: return None
        self.info['num_variables'] = A.shape[1]; self.info['num_samples'] = A.shape[0]
        if self.print_detial:
            print('Least Square samples:{}, variables:{}, ratio:{}'.format(
                self.info['num_samples'], self.info['num_variables'],
                round(self.info['num_samples']/self.info['num_variables'], 2)))
        self.w = np.linalg.lstsq(A, y, rcond=None)[0]
    def least_square(self, x, y, order=None, normalize=True, epoch=1, dropout=0.05):
        x = np.array(x); y = np.array(y)
        self.normalize = normalize
        self.order = order
        if self.normalize:
            y = self._scale_fit(y, name='y', eps=1e-4)
            x = self._scale_fit(x, name='x', eps=1e-4)
        self.x, self.y = x, y
        for tx in range(epoch):
            if tx != 0:
                yfit = self.forward(x, normalize=False)
            else:
                yfit = y.copy()
            x, y, yfit = self._apply_drop_outlier(x, y, yfit, rate=dropout)
            self._least_square(x, y)
    def forward(self, x, normalize=None):
        if normalize is None: normalize = self.normalize
        if normalize:
            x = self._scale_apply(x, name='x')
        A = self.convert2basis(x, self.order)
        y = 0
        for j in range(len(self.w)):
            y += self.w[j] * A[:, j]
        if normalize:
            y = self._scale_backward(y, name='y')
        return y
    def convert2basis(self, x, order=None):
        if order is None:
            print('Error! Wihout giving basis parameters')
            return None
        if x.ndim == 2:
            mode = 'multiple'; shape2 = x.shape[1]
        else:
            mode = 'single'
        A = []
        for ix in order:
            if mode == 'multiple':
                tmp = 1
                for vari in range(shape2):
                    tmp *= x[:, vari]**ix[vari]
            elif mode == 'single':
                tmp = x ** ix
            A += [tmp]
        A = np.vstack(A).T
        return A
    def print_equation(self, digits=3):
        nvar = len(self.order[0]) if isinstance(self.order[0], (list, tuple)) else 1
        terms = []
        for j in range(len(self.w)):
            ix = self.order[j] if isinstance(self.order[j], (list, tuple)) else [self.order[j]]
            coef = round(float(self.w[j]), digits)
            mono = []
            for k in range(len(ix)):
                e = ix[k]
                if e == 0: continue
                vname = 'x' if nvar == 1 else 'x{}'.format(k+1)
                mono.append(vname if e == 1 else '({}^{})'.format(vname, e))
            terms.append('{}*{}'.format(coef, '*'.join(mono)) if mono else '{}'.format(coef))
        eq = ('y = ' + ' + '.join(terms)).replace('+ -', '- ')
        if self.print_detial: print(eq)
        return eq
    def save(self, path='./pfit_weights.pkl'):
        data = {'w':self.w, 'order':self.order, 'normalize':self.normalize, 'scaler':self.scaler}
        save_pkl(path, data)
        return self
    def load(self, path='./pfit_weights.pkl'):
        data = load_pkl(path)
        self.w = data['w']; self.order = data['order']
        self.normalize = data['normalize']; self.scaler = data['scaler']
        return self
    def plot_linearity(self, x, y, xlabel='x', ylabel='y', save_name=None):
        nvar = len(self.order[0]) if isinstance(self.order[0], (list, tuple)) else 1
        exps = [ix[0] if isinstance(ix, (list, tuple)) else ix for ix in self.order]
        if nvar != 1 or set(exps) - {0, 1}:
            print('plot_linearity 僅適用單變數線性擬合 (order 僅含 0,1 次項)')
            return
        x = np.array(x); y = np.array(y)
        yfit = self.forward(x)
        err = np.abs(y - yfit)
        mean_err = err.mean(); max_err = err.max()
        eq = self.print_equation()
        plt.figure(figsize=(5, 4), dpi=120)
        plt.scatter(x, y, s=15, alpha=0.5, color='royalblue', label='Raw Data')
        plt.plot(x, yfit, color='crimson', lw=1.5, label='Linear Fit')
        plt.title('{}\nmean err = {:.4f},  max err = {:.4f}'.format(eq, mean_err, max_err), fontsize=9)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.tight_layout()
        if save_name is not None: plt.savefig(save_name)
        plt.show()

# ===== Experiment (not part of Poly_Fitting) =====
def fit_canny_with_user_class(image, epoch=1, dropout=0.0):
    h, w = image.shape
    edges = cv2.Canny(image, 50, 150)
    ys, xs = np.nonzero(edges)
    if len(xs) < 3:
        raise ValueError('Canny points are insufficient')

    # Select better-conditioned explicit representation.
    # x spread >= y spread: y = a + b*x; otherwise x = a + b*y.
    use_y_of_x = np.std(xs) >= np.std(ys)
    independent = xs.astype(float) if use_y_of_x else ys.astype(float)
    dependent = ys.astype(float) if use_y_of_x else xs.astype(float)

    pf = Poly_Fitting(print_detial=False)
    pf.least_square(independent, dependent, order=[0, 1],
                    normalize=True, epoch=epoch, dropout=dropout)

    # Recover unnormalized intercept and slope through the original forward().
    q = np.array([independent.min(), independent.max()], dtype=float)
    if q[0] == q[1]:
        raise ValueError('Independent coordinate has zero span')
    z = pf.forward(q)
    slope = (z[1] - z[0]) / (q[1] - q[0])
    intercept = z[0] - slope*q[0]

    if use_y_of_x:
        # y = intercept + slope*x -> slope*x - y + intercept = 0
        A, B, C = slope, -1.0, intercept
    else:
        # x = intercept + slope*y -> x - slope*y - intercept = 0
        A, B, C = 1.0, -slope, -intercept
    norm = np.hypot(A, B)
    A, B, C = A/norm, B/norm, C/norm

    # Orient the normal from black to white using image values, NOT truth phi.
    yy, xx = np.indices(image.shape)
    signed = A*xx + B*yy + C
    if signed[image >= 128].mean() < signed[image < 128].mean():
        A, B, C = -A, -B, -C

    cx, cy = w/2.0, h/2.0
    phi = np.degrees(np.arctan2(B, A)) % 360
    k = -(A*cx + B*cy + C) / min(w, h)
    residual = A*xs + B*ys + C
    line_rmse = np.sqrt(np.mean(residual**2))
    # Recompute directly from oriented coefficients for clarity.
    predicted = np.where(A*xx + B*yy + C >= 0, 255, 0).astype(np.uint8)
    image_rmse = np.sqrt(np.mean((image.astype(float)-predicted.astype(float))**2))
    return dict(pf=pf, edges=edges, xs=xs, ys=ys, A=A, B=B, C=C,
                phi=phi, k=k, line_rmse=line_rmse,
                image_rmse=image_rmse, predicted=predicted,
                representation='y(x)' if use_y_of_x else 'x(y)')


def main():
    W, H = 800, 600
    PHI_TRUE, K_TRUE = 120.0, 0.15
    EPOCH, DROPOUT = 1, 0.0  # change to 3, 0.05 for robust refitting
    phi = np.deg2rad(PHI_TRUE)
    yy, xx = np.indices((H, W))
    F = np.cos(phi)*(xx-W/2) + np.sin(phi)*(yy-H/2) - K_TRUE*min(W,H)
    image = np.where(F >= 0, 255, 0).astype(np.uint8)

    result = fit_canny_with_user_class(image, epoch=EPOCH, dropout=DROPOUT)
    angle_error = (result['phi']-PHI_TRUE+180) % 360 - 180
    print('Canny points:', len(result['xs']))
    print('Representation:', result['representation'])
    print(f"phi: true={PHI_TRUE:.6f}, fitted={result['phi']:.6f}, error={angle_error:+.6f} deg")
    print(f"k:   true={K_TRUE:.9f}, fitted={result['k']:.9f}, error={result['k']-K_TRUE:+.9f}")
    print(f"Line RMSE: {result['line_rmse']:.6f} pixels")
    print(f"Image RMSE: {result['image_rmse']:.6f} intensity units")
    print(f"Line: {result['A']:.9f}*x + {result['B']:.9f}*y + {result['C']:.9f} = 0")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('Original')
    axs[1].imshow(result['edges'], cmap='gray')
    axs[1].set_title(f"Canny: {len(result['xs'])} points")
    axs[2].imshow(image, cmap='gray', vmin=0, vmax=255)
    axs[2].scatter(result['xs'], result['ys'], s=0.5, c='red')
    A, B, C = result['A'], result['B'], result['C']
    midpoint = -C*np.array([A, B])
    direction = np.array([-B, A])
    t = np.linspace(-1200, 1200, 1000)
    line = midpoint[None, :] + t[:, None]*direction[None, :]
    axs[2].plot(line[:, 0], line[:, 1], 'c-', lw=1.5)
    axs[2].set_xlim(0, W); axs[2].set_ylim(H, 0)
    axs[2].set_title(f"Poly_Fitting: phi={result['phi']:.5f}, k={result['k']:.7f}")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig('single_canny_poly_result.png', dpi=160)
    plt.show()


if __name__ == '__main__':
    main()
