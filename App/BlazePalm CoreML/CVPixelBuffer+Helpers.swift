//
//  CVPixelBuffer+Helpers.swift
//  BlazePalm CoreML
//
//  Created by Vidur Satija on 17/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import Foundation
import Accelerate

func cropCGImage(_ inputImage: CGImage, xStart: Int, yStart: Int, width: Int, height: Int) -> CGImage? {
    let xFix = max(0, xStart)
    let yFix = max(0, yStart)
    let newWidth = min(width+xFix, inputImage.width)-xFix
    let newHeight = min(height+yFix, inputImage.height)-yFix
    let newRect = CGRect(x: xFix, y: yFix, width: newWidth, height: newHeight)
    return inputImage.cropping(to: newRect)
}
