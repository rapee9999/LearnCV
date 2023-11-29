"""
Utilities
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import requests
from PIL import Image
import io
import sys
import os
import typing



def getGithubFile(user: str, repo: str, branch: str, srcPath: str, dstPath: str = None) -> str:
    """Download a file from Github repo.

    Args:
        user (str): Github username.
        repo (str): Github repository name.
        branch (str): Repo's branch name.
        srcPath (str): Path to file in the repo.
        dstPath (str, optional): Local destination path to file. Defaults to None.

    Returns:
        str: Local destination path to file.
    """
    url = f"https://github.com/{user}/{repo}/raw/{branch}/{srcPath}"
    r = requests.get(url)
    if r.status_code != 200: raise RuntimeError(r)
    
    dstPath = dstPath or srcPath
    os.makedirs(os.path.dirname(dstPath), exist_ok=True)
    
    with Image.open(io.BytesIO(r.content)) as imgBytes:
        imgBytes.save(dstPath)
    
    return dstPath



def getCvSample(fileName: str, dstPath: str) -> str:
    """Download OpenCV's sample file.

    Args:
        fileName (str): File name in samples/data in OpenCV Github's repo.
        dstPath (str): Local path to file.

    Returns:
        str: Local path to file.
    """
    return getGithubFile("opencv", "opencv", "master", f"samples/data/{fileName}", dstPath)



def optimApproxPoly(vertice: np.ndarray, 
                    maxAreaDiff: float, 
                    startE: float, 
                    stepE: float, 
                    maxIter: int, 
                    verbose: bool = False) -> typing.Tuple[np.ndarray, float, float, float]:
    """Reduce vertices by approximating a contour until area reach maxinum area difference.

    Args:
        vertice (np.ndarray): An array of contour vertice.
        maxAreaDiff (float): Maximum area difference between original contour and approximated contour.
        startE (float): Starting coefficient to calculate approximation.
        stepE (float): Increment step of the coefficient.
        maxIter (int): Maximum number of iterations.
        verbose (bool, optional): Print results. Defaults to False.

    Returns:
        typing.Tuple[np.ndarray, float, float, float]: (Approximated contour vertice, last coefficient, last epsilon, last area different ratio)
    """
    # init.
    orgArcLen = cv2.arcLength(vertice, True)
    orgArea = cv2.contourArea(vertice)
    areaDiff = 0.0
    e = startE
    i = 0
    # optimize.
    while (areaDiff < maxAreaDiff and i < maxIter):
        eps = e * orgArcLen
        approx = cv2.approxPolyDP(vertice , eps, True)
        area = cv2.contourArea(approx)
        areaDiff = np.abs(area - orgArea) / orgArea
        e += stepE
        i += 1
    if verbose: 
        print(f"From area {orgArea} to {area}.")
        print(f"From vertice count {len(vertice)} to {len(approx)}.")
        print(f"e = {e}\neps = {eps}\narea diff = {areaDiff}")
    return approx, e, eps, areaDiff



def iterOptimApproxPoly(contours: typing.List[np.ndarray], 
                        maxAreaDiff: float, 
                        startE: float, 
                        stepE: float, 
                        maxIter: int, 
                        verbose: bool = False) -> list:
    """Iterate reducing vertices by approximating a contour until area reach maxinum area difference.

    Args:
        contours (typing.List[np.ndarray]): An array of contours.
        maxAreaDiff (float): Maximum area difference between original contour and approximated contour.
        startE (float): Starting coefficient to calculate approximation.
        stepE (float): Increment step of the coefficient.
        maxIter (int): Maximum number of iterations.
        verbose (bool, optional): Print results. Defaults to False.

    Returns:
        list: Approximated contours
    """
    approxContours = []
    for cnt in contours:
        oneStroke, _, _, _ = optimApproxPoly(cnt, maxAreaDiff, startE, stepE, maxIter, verbose)
        approxContours.append(oneStroke)
    return approxContours



def showContours(imageSize: typing.Tuple[int,int], 
                 contours: np.ndarray, 
                 title: str = "",
                 contourColor: typing.Tuple[int,int,int] = (255,255,255),
                 doShowIndice: bool = True,
                 indexStride: int = 1,
                 fontScale: float = 0.3,
                 fontThick: int = 1,
                 fontColor: typing.Tuple[int,int,int] = (255,255,255),
                 ax: plt.Axes = None,
                 doShow: bool = True) -> np.ndarray:
    """Show contours with vertice and index.

    Args:
        imageSize (typing.Tuple[int,int]): Image size height by width.
        contours (np.ndarray): Contours array.
        title (str, optional): Title shown above plot. Defaults to "".
        contourColor (typing.Tuple[int,int,int], optional): Color of contour lines and points. Defaults to (255,255,255).
        doShowIndice (bool, optional): Show point indice beside points. Defaults to True.
        indexStride (int, optional): Show index every indexStride. Defaults to 1.
        fontScale (float, optional): Font scale. Defaults to 0.3.
        fontThick (int, optional): Font thickness. Defaults to 1.
        fontColor (typing.Tuple[int,int,int], optional): Index font color. Defaults to (255,255,255).
        ax (plt.Axes, optional): Matplotlib's ax. Defaults to None.
        doShow: (bool, optional): Set to show image. Defaults to True.

    Returns:
        np.ndarray: Shown image.
    """
    image = np.zeros(list(imageSize) + [3], np.uint8)
    cv2.drawContours(image, contours, -1, contourColor, 1)
    for cnt in contours:
        for i, point in enumerate(cnt):
            cv2.circle(image, point[0], 1, contourColor, 2)
            if doShowIndice:
                if not (i % indexStride): 
                    cv2.putText(image, f"{i}", point[0], cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThick)
    if isinstance(ax, plt.Axes):
        ax.imshow(image), ax.set_title(title)
    elif doShow:
        plt.imshow(image), plt.title(title), plt.show()
    return image



def drawContourPlots(image: np.ndarray,
                     contours: np.ndarray,
                     hierarchy: np.ndarray,
                     ncols: int = 3,
                     figScale: typing.Tuple[int,int] = (4,4),
                     imShow: bool = True,
                     showColor: bool = False) -> np.ndarray:
    """Breakdown contours to subplots. Each subplot draw each contour.

    Args:
        image (np.ndarray): Image which contours are found.
        contours (np.ndarray): A list of contours.
        hierarchy (np.ndarray): A list of contour hierarchy.
        ncols (int, optional): Number of subplot columns. Defaults to 3.
        figScale (Tuple[int,int], optional): Figure size for each contour. Defaults to (4,4).
        imShow (bool, optional): Set to display or otherwise. Defaults to True.
        showColor (bool, optional): Set to display color. Defaults to False.

    Returns:
        np.ndarray: Images of individual contours.
    """
    cntNum = len(contours)
    cntImgs = np.array([], np.uint8).reshape(0, *image.shape)
    
    if imShow:
        ncols = np.min([ncols, cntNum])
        nrows = int(np.ceil(cntNum / ncols)) + 1
        fig = plt.figure(figsize=(ncols * figScale[0], nrows * figScale[1]))
        gs = fig.add_gridspec(nrows, ncols)
        
        ax = fig.add_subplot(gs[0, :])
        ax.imshow(image, 'gray')
        ax.set_title("Input")
        
    for i in range(cntNum): 
        img = np.zeros(image.shape, np.uint8)
        img = cv2.drawContours(img, contours, i, 255, 1)
        cntImgs = np.append(cntImgs, img.reshape(1, *img.shape), 0)
        if imShow: 
            ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
            ax.set_title(f"{i} {hierarchy[0][i]}")
            if showColor:
                fillMap = img == 255
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img[fillMap] = (int(i*360/cntNum),255,255)
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
                ax.imshow(img)
            else:
                ax.imshow(img, 'gray')
    if imShow:
        fig.show()
        
    return cntImgs

