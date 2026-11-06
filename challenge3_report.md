# Lab 3 Edge Challenge: report

**Student:** Alaa Abdelhalim Khamis   
**Student ID:** 231015225  
**Challenge 3:** Garage camera at night  
**Date:** 2026-11-06

## The image

A security camera recorded a motorcycle parked in a dark garage. Almost nothing is visible and the sensor has added strong, coloured grain.

Problems: Very dark: most values are below 40; Strong colour noise (shot noise plus sensor noise, different in each channel).

Target: Outline the motorcycle, its wheels, and the shelves and boxes behind it.

## Main pipeline

F1 **84.1** (precision 85.7, recall 82.5), Level 3. Baseline 63.8, level 3 at 82.9.

Steps: gray > gaussian (k=9, s=1.2) > equalize > Prewitt > percentile (q=78)

```python
import cv2, numpy as np

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32); SOBEL_Y = SOBEL_X.T.copy()
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32); PREWITT_Y = PREWITT_X.T.copy()
ROBERTS_X = np.array([[1, 0], [0, -1]], np.float32); ROBERTS_Y = np.array([[0, 1], [-1, 0]], np.float32)

img = load(3)   # challenge 3: Garage camera at night

# Colour to one channel
x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Smoothing
x = cv2.GaussianBlur(x, (9, 9), 1.2)

# Contrast
x = cv2.equalizeHist(x)

# Edge detection
gx = cv2.filter2D(x, cv2.CV_32F, PREWITT_X)
gy = cv2.filter2D(x, cv2.CV_32F, PREWITT_Y)
x = cv2.normalize(np.sqrt(gx**2 + gy**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Thresholding
x = ((x > np.percentile(x, 78)) * 255).astype(np.uint8)

edges = x
```

**Why each step is there**

My main pipeline is: grayscale, Gaussian blur (kernel 9, sigma 1.2), equalizeHist, Prewitt gradient magnitude, and a percentile threshold at q = 78. It scores F1 84.1 (precision 85.7, recall 82.5), which is Level 3.

The image has two problems: it is very dark (most values below 40) and it has strong colour noise that is different in each channel. Converting to grayscale averages the three channels, which reduces this independent noise. Any single channel scored 6.9 to 9.1 F1 lower. Gaussian smoothing suppresses the remaining grain so it does not produce false gradients. Without it, precision fell from 85.7 to 62.7. equalizeHist spreads the dark histogram so real edges have enough gradient strength to pass the threshold. Prewitt computes the gradient, and the percentile threshold adapts to the weak gradient values of a dark image instead of using a fixed number. I left morphology out because it removed thin real edges such as wheel spokes and never improved the score.

## Ablation study

### Leave one out

| Variant | Change | F1 | Δ F1 | Precision | Recall |
|---|---|---|---|---|---|
| Main pipeline | - | 84.1 | 0.0 | 85.7 | 82.5 |
| without edge detection | remove Prewitt | 15.9 | -68.2 | 15.4 | 16.4 |
| without smoothing | remove gaussian | 73.9 | -10.2 | 62.7 | 90.2 |
| without contrast | remove equalize | 80.0 | -4.1 | 83.0 | 77.1 |

**Interpretation:** Removing edge detection dropped F1 by 68.2 (84.1 to 15.9), so it is the most important step: without a gradient map the percentile threshold has nothing meaningful to act on. Removing Gaussian smoothing dropped F1 by 10.2 and precision fell from 85.7 to 62.7 while recall rose to 90.2. The strong colour grain creates many false gradients, and smoothing suppresses them. Removing equalizeHist cost 4.1 points, mainly in recall (82.5 to 77.1), because the very dark image has weak gradients that contrast enhancement lifts above the threshold. Smoothing therefore matters mostly for precision and contrast mostly for recall 

### Swap the method

| Variant | Change | F1 | Δ F1 | Precision | Recall |
|---|---|---|---|---|---|
| Main pipeline | - | 84.1 | 0.0 | 85.7 | 82.5 |
| Smoothing: mean | replace gaussian with mean | 60.5 | -23.6 | 59.7 | 61.4 |
| Morphology: erode | add erode | 62.6 | -21.5 | 96.7 | 46.3 |
| Smoothing: median | replace gaussian with median | 65.5 | -18.6 | 66.3 | 64.7 |
| Morphology: open | add open | 69.2 | -14.9 | 90.8 | 55.9 |
| Colour to one channel: blue | replace gray with blue | 75.0 | -9.1 | 72.9 | 77.1 |
| Colour to one channel: red | replace gray with red | 75.4 | -8.7 | 74.1 | 76.8 |
| Colour to one channel: green | replace gray with green | 77.2 | -6.9 | 74.9 | 79.7 |
| Morphology: dilate | add dilate | 80.0 | -4.1 | 71.7 | 90.3 |
| Contrast: stretch | replace equalize with stretch | 80.2 | -3.9 | 82.6 | 78.0 |
| Sharpening: sharpen kernel | add sharpen kernel | 81.0 | -3.1 | 74.9 | 88.1 |
| Contrast: gamma | replace equalize with gamma | 81.7 | -2.4 | 83.0 | 80.4 |
| Edge detection: Roberts | replace Prewitt with Roberts | 82.2 | -1.9 | 78.6 | 86.1 |
| Morphology: close | add close | 82.6 | -1.5 | 82.3 | 82.8 |
| Sharpening: unsharp mask | add unsharp mask | 83.4 | -0.7 | 81.6 | 85.3 |
| Edge detection: Sobel | replace Prewitt with Sobel | 84.0 | -0.1 | 85.6 | 82.4 |
| Thresholding: fixed T | replace percentile with fixed T | 84.3 | +0.2 | 84.6 | 84.0 |
| Thresholding: Otsu | replace percentile with Otsu | 84.4 | +0.3 | 81.6 | 87.4 |

**Interpretation:** Replacing Gaussian with mean or median dropped F1 by 23.6 and 18.6, so the choice of smoothing filter matters more than the choice of edge operator (Roberts -1.9, Sobel -0.1). Gaussian suppresses the colour sensor noise while keeping thin edges. Erode and open dropped F1 by 21.5 and 14.9 because recall fell to 46.3 and 55.9: they remove thin real edges. Any single colour channel lost 6.9 to 9.1 points against grayscale, since averaging the channels reduces independent noise. Otsu (+0.3) and fixed T (+0.2) matched the percentile, so once smoothing and contrast are right, the threshold method is not critical.

### Parameter sweep: Thresholding: percentile percentile q

| Variant | Change | F1 | Δ F1 | Precision | Recall |
|---|---|---|---|---|---|
| Main pipeline | - | 84.1 | 0.0 | 85.7 | 82.5 |
| q = 50 | q = 50 | 76.5 | -7.6 | 62.7 | 98.0 |
| q = 52 | q = 52 | 77.1 | -7.0 | 63.7 | 97.7 |
| q = 54 | q = 54 | 78.1 | -6.0 | 65.3 | 97.2 |
| q = 56 | q = 56 | 78.8 | -5.3 | 66.4 | 96.8 |
| q = 58 | q = 58 | 80.2 | -3.9 | 68.8 | 96.0 |
| q = 60 | q = 60 | 80.7 | -3.4 | 69.7 | 95.7 |
| q = 62 | q = 62 | 81.5 | -2.6 | 71.5 | 94.8 |
| q = 64 | q = 64 | 82.2 | -1.9 | 73.1 | 93.9 |
| q = 66 | q = 66 | 82.7 | -1.4 | 74.3 | 93.3 |
| q = 68 | q = 68 | 83.3 | -0.8 | 76.1 | 92.1 |
| q = 70 | q = 70 | 84.1 | 0.0 | 78.8 | 90.1 |
| q = 72 | q = 72 | 84.2 | +0.1 | 80.1 | 88.8 |
| q = 74 | q = 74 | 84.4 | +0.3 | 82.4 | 86.6 |
| q = 76 | q = 76 | 84.4 | +0.3 | 83.9 | 85.0 |
| q = 78 (main) | q = 78 | 84.1 | 0.0 | 85.7 | 82.5 |
| q = 80 | q = 80 | 83.4 | -0.7 | 87.7 | 79.4 |
| q = 82 | q = 82 | 82.4 | -1.7 | 89.3 | 76.5 |
| q = 84 | q = 84 | 80.3 | -3.8 | 91.0 | 71.8 |
| q = 86 | q = 86 | 78.3 | -5.8 | 92.7 | 67.8 |
| q = 88 | q = 88 | 74.6 | -9.5 | 94.2 | 61.7 |
| q = 90 | q = 90 | 69.5 | -14.6 | 95.5 | 54.7 |
| q = 92 | q = 92 | 63.4 | -20.7 | 96.7 | 47.1 |
| q = 94 | q = 94 | 56.4 | -27.7 | 97.5 | 39.6 |
| q = 96 | q = 96 | 45.2 | -38.9 | 98.2 | 29.3 |
| q = 98 | q = 98 | 28.5 | -55.6 | 99.1 | 16.7 |

**Interpretation:** Varying the Gaussian kernel size showed a plateau. From k = 7 to k = 15 the scores were identical (F1 84.1), because with sigma fixed at 1.2 the kernel weights beyond about 7 pixels are close to zero, so a larger window adds no smoothing. Only k = 3 differed (F1 82.7, -1.4): the window was too small to suppress the colour grain, so precision fell to 79.6 while recall rose to 86.0. This shows that the amount of smoothing is set by sigma, and kernel size only needs to be large enough to hold the Gaussian

### Step order

| Variant | Change | F1 | Δ F1 | Precision | Recall |
|---|---|---|---|---|---|
| Main pipeline | - | 84.1 | 0.0 | 85.7 | 82.5 |
| contrast > smoothing | order: contrast > smoothing | 83.7 | -0.4 | 86.4 | 81.1 |
| smoothing > contrast (main) | order: smoothing > contrast | 84.1 | 0.0 | 85.7 | 82.5 |

**Interpretation:** Running contrast before smoothing gave F1 83.7 against 84.1 for the main order (-0.4), with precision 86.4 and recall 81.1. The difference is small, so the order of these two steps has little effect on this image. Smoothing first is slightly better because equalizeHist stretches the histogram of the dark image and enlarges the colour grain along with the signal. If the grain is removed first, contrast enhancement acts on a cleaner image. Since sharpening is not used, only two orders exist.

## Discussion

**Which step contributes most and least**

Edge detection contributes most. Removing it dropped F1 from 84.1 to 15.9 (-68.2), because without a gradient map the threshold has nothing meaningful to act on. Among the cleaning steps, smoothing contributes most: removing the Gaussian dropped F1 by 10.2 (to 73.9), and precision fell from 85.7 to 62.7 because grain created false edges.
Contrast contributes least of the steps in the pipeline. Removing equalize cost only 4.1 F1 (to 80.0), mainly through recall (82.5 to 77.1). Morphology was the least useful step overall, since adding it never helped. Erode and open lowered F1 by 21.5 and 14.9, and close lowered it by 1.5. The choice of edge operator also barely mattered: Sobel was -0.1 and Roberts -1.9 compared with Prewitt.

**Most sensitive setting**

The percentile threshold q was the most sensitive setting. Sweeping q from 50 to 98 moved F1 from about 76 at q = 50 up to a peak of 84.4 at q = 74, then down to about 28 at q = 98. Precision rises steadily with q (from about 62 to about 99) while recall falls (from about 98 to about 17). The two curves cross near q = 75 to 77, where F1 is highest.
q decides what fraction of gradient pixels is kept as edges: the top (100 - q)%. A low q keeps weak gradients, including grain, so recall is high but precision is low. A high q keeps only the strongest gradients, so what remains is mostly real edge (precision high), but weak true edges such as thin spokes and shelf lines are thrown away (recall collapses). My main pipeline uses q = 78, within 0.3 F1 of the best value (q = 74, F1 84.4), so I kept it. By comparison, the Gaussian kernel size sweep was flat from k = 7 to k = 15 (F1 84.1 throughout), because with sigma fixed at 1.2 a larger window adds no extra blur. Only k = 3 differed (F1 82.7)

**Limits and next steps**

Gaussian smoothing reduces noise but also blurs real edges, so there is a trade-off between a clean dark floor and thin structures like wheel spokes and the detail in the cluttered shelves. A single global threshold cannot adapt to local differences in brightness and noise, so one q cannot be ideal for both the bright motorcycle and the dark background. Edge detection also produces thick, broken edge lines, because Prewitt and Sobel give no edge thinning and no way to connect weak edge pixels to strong ones.
Next I would try three things. First, denoising that preserves edges, such as bilateral filtering or non-local means, before the gradient step. Second, Canny edge detection with non-maximum suppression and hysteresis thresholding, to thin edges and keep weak edges that connect to strong ones. Third, local (adaptive) thresholding, so the threshold follows the local brightness.

## Experiment log

| # | F1 | Precision | Recall | Pipeline | Note |
|---|---|---|---|---|---|
| 1 | 80.3 | 78.4 | 82.3 | gray > gamma (g=0.6) > gaussian (k=5, s=1) > Sobel > percentile (q=79.5) > close (rect 3x3) | median to Gaussian σ=1, q=79.5, F1 80.3 |
| 2 | 81.6 | 82.1 | 81.0 | gray > gamma (g=0.6) > gaussian (k=5, s=1.1) > Sobel > percentile (q=79.5) > close (cross 3x3) | cross 3×3 + MORPH_CLOSE, q about 79.5, F1 81.6 |
| 3 | 63.7 | 79.6 | 53.0 | gray > gamma (g=0.6) > gaussian (k=5, s=1.1) > Sobel > percentile (q=92.5) > close (ellipse 9x9) | q 92.5 + ellipse 9×9: F1 81.6 to 63.7, recall collapsed |
| 4 | 82.1 | 79.5 | 84.8 | gray > gamma (g=0.6) > gaussian (k=5, s=1.1) > Sobel > percentile (q=77.5) > close (cross 3x3) | q 77.5, cross 3×3 + CLOSE: F1 82.1, precision 79.5, recall 84.8 |
| 5 | 83.8 | 83.2 | 84.5 | gray > gamma (g=0.6) > gaussian (k=5, s=1.1) > Sobel > percentile (q=77.5) | removed morphology, q 77.5: F1 83.8, Level 3, precision 83.2, recall 84.5 |
| 6 | 83.4 | 80.6 | 86.3 | gray > mean (k=3) > equalize > Prewitt > percentile (q=77.5) | equalize + Prewitt |
| 7 | 83.5 | 81.5 | 85.5 | gray > mean (k=3) > equalize > Prewitt > percentile (q=78) | q = 78 |
| 8 | 84.1 | 85.7 | 82.5 | gray > gaussian (k=9, s=1.2) > equalize > Prewitt > percentile (q=78) | blur 9×9 σ1.2, then equalizeHist, Prewitt, q 78: F1 84.1, P 85.7, R 82.5 |