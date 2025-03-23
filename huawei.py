""""""
from pathlib import WindowsPath
from dataclasses import dataclass

import cv2
import easyocr


@dataclass
class PicRange:
    x_start:int
    x_end:int
    y_start:int
    y_end:int

    
class HuaweiHealthCareOCR:
    def __init__(self,fp:str|WindowsPath):
        self.fp = fp

    def _lim_img(self,
                 pic_range:PicRange):
        img = cv2.imread(self.fp) 
        # img.shape = (3261, 720, 3)
        cpd = img[
            pic_range.y_start:pic_range.y_end,
            pic_range.x_start:pic_range.x_end,
        ]
        cv2.imwrite("crp.jpg",cpd)

    def _lim_ocr(self,
                 pic_range:PicRange,
                 *,
                 allowlist:list=None):
        img = cv2.imread(self.fp)
        cpd = img[
            pic_range.y_start:pic_range.y_end,
            pic_range.x_start:pic_range.x_end,
        ]
        reader = easyocr.Reader(["ja"],gpu=True)
        return reader.readtext(
                cpd,
                detail=0,
                allowlist=allowlist,
                decoder="greedy",
            )


