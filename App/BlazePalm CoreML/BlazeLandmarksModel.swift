//
//  BlazeLandmarksModel.swift
//  BlazePalm CoreML
//
//  Created by Vidur Satija on 17/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import Foundation
import Vision
import CoreML
import Accelerate
import CoreGraphics
import CoreImage
import VideoToolbox

class BlazeLandmarksModelInput: MLFeatureProvider {
    private static let imageFeatureName = "image"

    var imageFeature: CGImage

    var featureNames: Set<String> {
        return [BlazeLandmarksModelInput.imageFeatureName]
    }

    init(image: CGImage) {
        imageFeature = image
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == BlazeLandmarksModelInput.imageFeatureName else {
            return nil
        }

        let options: [MLFeatureValue.ImageOption: Any] = [
            .cropAndScale: VNImageCropAndScaleOption.scaleFit.rawValue
        ]

        return try? MLFeatureValue(cgImage: imageFeature,
                                   pixelsWide: 256,
                                   pixelsHigh: 256,
                                   pixelFormatType: imageFeature.pixelFormatInfo.rawValue,
                                   options: options)
    }
}

class BlazeLandmarksBatchInput: MLBatchProvider {
    var batchArray: [MLFeatureProvider?]
    var pos: Int
    var count: Int
    init(batch: Int) {
        pos = 0
        count = batch
        batchArray = [MLFeatureProvider?](repeating: nil, count: batch)
    }

    func append(xFeature: MLFeatureProvider) {
        batchArray[pos] = xFeature
        pos += 1
    }

    func features(at index: Int) -> MLFeatureProvider {
        return batchArray[index]!
    }
}
