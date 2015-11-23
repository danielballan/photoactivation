# Notes

## 2015-11-23 at Penn

### Water

Water is reasonably robust for different details of the analysis procedure. 

* We tried fixing the height of the Gaussian in the model based on the change in intensity. This was actually wrong, but the diffusivities came out *better*.
* Then we fixed the *amplitude* of the Gaussian, which is correct, and diffusivities only got slightly worse.
* We tried going through the mappings in reverse order -- still mapping "forward" but starting from the last interval in the video and working backward. This has a negligible but nonzero affect on the sigma.

The water figure looks good.
   
### Fibrin

Fibrin analysis varies from water analysis in some ways that still should be better understood.
* Subtraction of the "stuck map" must be done pixelwise because the subtraction is nonlienar due to clipping at zero.
* Subtraction of "stuck map" must be done as int to correspond to my old results (which are better). No idea why floats don't give the same result in this case.
* Before mapping, all must be shifted by a constant so that their min = 0. Empirically, the results are worse if I don't do this.
* Using a bound on the fit (`bound=True`) improves the result in this case. It does not improve the result for water.

What will the figure look like?

* Include the Gaussian mobile/immobile plot. Change the color from red -- red is the activation region color.
* Show individual MDS or two ensemble MSDs. We have three Fibrin MSD results, made on different samples by different people. Expected D~1.5 from MPT (mobile fraction). As of this writing, the D from PANDA (mobile fraction) is about 1.0 +/- 0.05.
* In the four windows of profiles, definitely show the unsubtracted. Maybe also show subtracted.

### CFS

CFS is difficult. The following things had only a small effect on the sigma. Unlike the other samples, the CFS video has a consistent pathology not affected by 
* trying reverse order (as with water)
* fixing height (mistakenly) and then amplitude (correctly) as with water

Without amplitude fixed, the nonlinear fitting finds a way to produce a good fit, but it results to an unphysically small amplitude and a negative baseline to achieve it. With the amplitude fixed, the mappings for late frames are very poor, as is visually obvious. Ideas:
* Surrender and just map the first frame to the last (if indeed that mappings looks good) and extract a single number for D.
* Mask out the shaded areas (as NaN) and compute a profile *average* (not sum).
* Look at other videos.

What will Figure 5 look like? Some ideas, pending more analysis:
* just like Figure 3
* include sigma from multiple different samples/patients on one or multiple plots
* a histogram of D from particle tracking?
* a histogram of D from simplified (single-mapping) PANDA?

### Supplemental Material

* a couple real mappings, as in the interactive mapping explorer
* the mottled view of fibrin
* a rainbow plot, or multiple
* galleries and, if possible, videos
* the table of particles sizes and zeta potential measured by DLS
* a figure showing MSD from MPT experiments of PS-PEG and PS-COOH
* linearity of fluorescence vs. concentration
* simulation video, if we use it in the end