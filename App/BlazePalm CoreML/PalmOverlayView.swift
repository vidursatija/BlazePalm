//
//  PalmOverlayView.swift
//  BlazePalm CoreML
//
//  Created by Vidur Satija on 07/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import Foundation

import UIKit
import Vision

class PalmOverlayView: UIView {

    var landmarks: [Palm] = []

    func clear() {
        landmarks = []
    }
    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else {
          return
        }
        context.saveGState()
        defer {
          context.restoreGState()
        }

        for palm in landmarks {
            for ith in 0..<21 {
                context.addEllipse(in:
                                    CGRect(x: CGFloat(palm.landmarks[ith]!.x)*self.frame.size.width-2.5,
                                           y: CGFloat(palm.landmarks[ith]!.y)*self.frame.size.height-2.5,
                                           width: 5.0,
                                           height: 5.0)
                )
                UIColor.init(hue: CGFloat(2*palm.landmarks[ith]!.z+1),
                             saturation: 1.0,
                             brightness: 1.0,
                             alpha: 1.0)
                    .setStroke()
                context.strokePath()
            }
        }

    }
}
