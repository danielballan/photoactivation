from lmfit.models import ConstantModel, GaussianModel
from functools import lru_cache
from lmfit.model import Model
from numpy import convolve, exp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm


def make_profiler(background, axis):
    """
    Build a function that subtracts the background and sums along an axis
    
    Parameters
    ----------
    background : number
        intensity to subtract from the profile
    axis : int
        which axis to sum along
        
    Returns
    -------
    func :
        function with the signature (img_stack, i)
    """
    @lru_cache(maxsize=1001)
    def func(img_stack, i):
        # Sum and convert to float64, which is safe in subtraction.
        raw_profile = np.asarray(np.sum(img_stack[i], axis)).astype(np.float64)
        return np.clip(raw_profile - background, 0, None)
    return func


## Curve Fitting ##                                  
                                
class ConvolvedGaussianModel(Model):
    def __init__(self, kernel, margin=50, **kwargs):
        assert margin < len(kernel) // 2 and margin > 0, "User is a doofis."
        self.kernel = kernel
        self.margin = margin
        
        def func(x, height, sigma, center, base):
            # Note: This uses kernel, defined just above.
            gaussian = base + height * exp(-(x-center)**2/(2*sigma**2))
            result = convolve(gaussian, self.kernel, 'same')
            return result[self.margin:-self.margin]
        
        super(ConvolvedGaussianModel, self).__init__(func, **kwargs)
        
    def guess_starting_values(self, profile):
        self.set_param_hint('height', value=1, min=0)
        self.set_param_hint('sigma' , value=len(profile) / 2 - np.argmax(profile - profile.min() > profile.ptp() / 2),
                            min=0)
        self.set_param_hint('center', value=len(profile) / 2, min=0, max=len(profile))
        self.set_param_hint('base', value=0, min=0)
        self.has_initial_guess = True
    
    def fit(self, profile, **kwargs):
        # TODO Justify margin=50 once I understand what the word convolution means.
        profile = profile.copy()
        m = self.margin  # for brevity
        x = np.arange(m, 512 - m)

        if not self.param_hints:
            self.guess_starting_values(profile[m:-m])
            
        fit_result = super(ConvolvedGaussianModel, self).fit(profile[m:-m], x=x, **kwargs)

        return fit_result
    

def fit_line(x, y):
    """Return slope, intercept of best fit line."""
    # intercept is forced to zero
    model = sm.OLS(y, x, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    return fit


def fit_profiles_recursively(profiles, lag, *, recursive=True, bound=False):
    """
    Parameters
    ----------
    profiles: list
        profiles evenly spaced in time
    lag : int
        profiles to skip
    recursive : bool, optional
        If True, use last fit as initial guess for current fit.
        True by default.
    bound : bool, optional
        If True, apply min/max bounds to the fit.
        False by default.
        
    Returns
    -------
    results : list
        list of lmfit Result objects
    """
    kernels = [profile - np.min(profile)
               for profile in profiles[:-lag]]

    results = []
    for kernel, profile in zip(kernels, profiles[lag:]):
        cg_model = ConvolvedGaussianModel(kernel)

        # If this is not our first iteration, use the best fit from the last
        # point as our initial guess for this point. Otherwise, the model
        # knows how to generated an automated guess.
        if recursive and results:
            for name, p in results[-1].params.items():
                # Specificying min and max breaks the fit.
                if bound:
                    cg_model.set_param_hint(name, value=p.value, min=p.min,
                                            max=p.max)
                else:
                    cg_model.set_param_hint(name, value=p.value)

        results.append(cg_model.fit(profile))
    return results


def fit_profile(profile, guess):
    "Fit a profile to a Gaussian + Contant"
    x = np.arange(len(profile))

    model = GaussianModel(missing='drop') + ConstantModel(missing='drop')
    result = model.fit(profile, x=x, verbose=False, **guess)
    return result


def fit_profile_tails(profile, guess, halfwidth, center=None):
    if center is None:
        center = len(profile) // 2
    
    profile_tails = profile.copy()
    profile_tails[center - halfwidth: center + halfwidth] = np.nan
    return fit_profile(profile_tails, guess)


def extrapolate_fit(tails_result, x):
    full_gaussian = tails_result.eval(x=x)
    return full_gaussian


## Plots ##


def plot_tail_fit(profile, tails_result, halfwidth, center=None):
    if center is None:
        center = len(profile) // 2

    fig, ax = plt.subplots()
    x = np.arange(len(profile))
    x1, x2 = (np.arange(center - halfwidth),
              np.arange(center + halfwidth, len(profile)))
    profile_tails = tails_result.data
    full_gaussian = tails_result.eval(x=x)  # extrapolating through center
    ax.plot(x, profile, color='gray')
    ax.plot(x1, profile_tails[:center - halfwidth], color='black')
    ax.plot(x2, profile_tails[center - halfwidth:], color='black')
    ax.plot(x, full_gaussian, color='red')
    

def outline_activation_region(img):
    fig, ax = plt.subplots()
        
    IMAGE_SIZE = 512
    STRIPE_WIDTH = 40
    dimensions = ((0, IMAGE_SIZE//2 - STRIPE_WIDTH/2), IMAGE_SIZE - 1, STRIPE_WIDTH)
    style = dict(facecolor='none', edgecolor='red', linewidth=2)
    
    ax.imshow(img)
    rectangle = mpl.patches.Rectangle(*dimensions,
                                      transform=ax.transData,
                                      **style)
    ax.add_patch(rectangle)


## Image utilities

def subtract_safely(a, b):
    """
    Avoid wrap-around issue with integer subtraction, converting <0 to 0.
    
    Example
    -------
    # This is bad.
    >>> a = np.array([5], 'uint16')
    >>> b = np.array([4], 'uint16')
    >>> b - a
    array([65535], dtype=uint16)
    
    # This is good.
    >>> subtract_safely(b, a)
    array([0], dtype=uint16)
    """
    return np.clip(a.astype('float64') - b, 0, None)


def line_as_image(line, width):
    return np.repeat(np.expand_dims(line / width, 1), width, 1).astype('uint16')
