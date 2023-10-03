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
