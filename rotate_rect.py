import cv2
import numpy as np


if __name__ == "__main__":
    print("Running...")
    
    # init.
    imgSize = np.array([500,500], np.int32) # width, height
    center = np.array([250,250], np.float16) # x, y
    size = np.array([100, 200], np.float16) # width, height (major, minor)
    dreg = 0
    interval = 25
    
    pause = False
    while True:
        if not pause:
            # find major and minor axises.
            MA = np.array([np.cos(np.radians(dreg)) * size[0]/2, np.sin(np.radians(dreg)) * size[0]/2])
            ma = np.array([np.cos(np.radians(dreg + 90)) * size[1]/2, np.sin(np.radians(dreg + 90)) * size[1]/2])
            # init rect.
            rect = cv2.RotatedRect(center, size, dreg)
            points = cv2.boxPoints(rect)
            # empty image.
            img = np.zeros(np.flip(imgSize), np.uint8)
            # rotation anotation.
            cv2.ellipse(img, center.astype(np.int32), (50, 50), 0, 0, dreg, 25, cv2.FILLED)
            # guide lines.
            cv2.line(img, (center[0].astype(np.int32), 0), (center[0].astype(np.int32), imgSize[1]), 50, 1)
            cv2.line(img, (0, center[1].astype(np.int32)), (imgSize[0], center[1].astype(np.int32)), 50, 1)
            cv2.arrowedLine(img, center.astype(np.int32), (center[0].astype(np.int32), imgSize[1]-10), 100, 1, tipLength=0.04)
            cv2.arrowedLine(img, center.astype(np.int32), (imgSize[0]-10, center[1].astype(np.int32)), 100, 1, tipLength=0.04)
            cv2.putText(img, "y", (center[0].astype(np.int32)+10, imgSize[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 100, 1)
            cv2.putText(img, "x", (imgSize[0]-20, center[1].astype(np.int32)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 100, 1)
            # ellipse
            cv2.ellipse(img, rect, 255, 1)
            # bounding box.
            cv2.polylines(img, [points.astype(np.int32)], isClosed=True, color=100, thickness=1)
            for i, p in enumerate(points.astype(np.int32)):
                cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 100, 1)
            # major and minor axises.
            cv2.arrowedLine(img, center.astype(np.int32), (center + MA).astype(np.int32), 255, 1, tipLength=0.1)
            cv2.arrowedLine(img, center.astype(np.int32), (center + ma).astype(np.int32), 255, 1, tipLength=0.1)
            cv2.putText(img, "MAJOR", (center + MA).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            cv2.putText(img, "minor", (center + ma).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
            # rotation anotation.
            cv2.circle(img, center.astype(np.int32), 2, 255, 2)
            cv2.putText(img, f"{int(dreg)}", (center + np.array([np.cos(np.radians(dreg/2)) * size[0]/2, np.sin(np.radians(dreg/2)) * size[0]/2])).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            # show.
            cv2.imshow("Rotate", img)
        k = cv2.waitKey(interval)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):
            pause = not pause
        elif k == ord(']'):
            if interval > 10:
                interval -= 5
        elif k == ord('['):
            if interval < 100:
                interval += 5
        # next
        if not pause:
            dreg += 1
            dreg %= 360   
            
    print("End!")
    