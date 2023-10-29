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
                 doShowIndice: bool = True, 
                 fontScale: float = 0.3,
                 fontThick: int = 1,
                 axes: plt.Axes = None,
                 doShow: bool = True) -> np.ndarray:
    """Show contours with vertice and index.

    Args:
        imageSize (typing.Tuple[int,int]): Image size height by width.
        contours (np.ndarray): Contours array.
        title (str, optional): Title shown above plot. Defaults to "".
        doShowIndice (bool, optional): Show point indice beside points. Defaults to True.
        fontScale (float, optional): Font scale. Defaults to 0.3.
        fontThick (int, optional): Font thickness. Defaults to 1.
        axes (plt.Axes, optional): Matplotlib's axes. Defaults to None.
        doShow: (bool, optional): Set to show image. Defaults to True.

    Returns:
        np.ndarray: Shown image.
    """
    image = np.zeros(list(imageSize) + [3], np.uint8)
    cv2.drawContours(image, contours, -1, (255,255,255), 1)
    for cnt in contours:
        for i, point in enumerate(cnt):
            cv2.circle(image, point[0], 1, (255,255,255), 2)
            if doShowIndice: cv2.putText(image, f"{i}", point[0], cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255,255,255), fontThick)
    if isinstance(axes, plt.Axes):
        axes.imshow(image), axes.set_title(title)
    elif doShow:
        plt.imshow(image), plt.title(title), plt.show()
    return image

