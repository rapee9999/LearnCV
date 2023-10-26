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
