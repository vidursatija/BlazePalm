//
//  BlazePalmModel.swift
//  BlazePalm CoreML
//
//  Created by Vidur Satija on 07/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import Foundation
import Vision
import CoreML
import Accelerate
import CoreGraphics
import CoreImage
import VideoToolbox

class BlazePalmModelInput: MLFeatureProvider {
    private static let imageFeatureName = "image"

    var imageFeature: CGImage

    var featureNames: Set<String> {
        return [BlazePalmModelInput.imageFeatureName]
    }

    init(image: CGImage) {
        imageFeature = image
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard featureName == BlazePalmModelInput.imageFeatureName else {
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

public func IOU(_ rect1: SIMD4<Float32>, _ rect2: SIMD4<Float32>) -> Float {
    let areaA = (rect1[3]-rect1[1]) * (rect1[2]-rect1[0])
    if areaA <= 0 { return 0 }

    let areaB = (rect2[3]-rect2[1]) * (rect2[2]-rect2[0])
    if areaB <= 0 { return 0 }

    let intersectionMinX = max(rect1[1], rect2[1])
    let intersectionMinY = max(rect1[0], rect2[0])
    let intersectionMaxX = min(rect1[3], rect2[3])
    let intersectionMaxY = min(rect1[2], rect2[2])
    let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                            max(intersectionMaxX - intersectionMinX, 0)
    return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

class Palm {
    var landmarks: [SIMD4<Float>?]
    var boundingBox: CGRect
    var confidence: Float

    init(landmarks: [SIMD4<Float>?], bBox: CGRect, conf: Float) {
        self.landmarks = landmarks
        self.boundingBox = bBox
        self.confidence = conf
    }
}

class BlazePalmModel {
    var model: MLModel?
    var landmarkModel: MLModel?
    let minConfidence: Float32 = 0.75
    let nmsThresh = 0.3
    let maxPalms = 2
    let dimScale = Float(1.5)
    var previousCenters: [SIMD4<Float32>] = [] // x,y,width,height

    init() {
        self.model = BlazePalmScaled().model
        self.landmarkModel = try? BlazeLandmarks().model
    }

    func predict(for buffer: CVPixelBuffer) -> [Palm] {
        var imageFeature: CGImage?
        VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &imageFeature)
        let imgH = Float32(imageFeature!.height)
        let imgW = Float32(imageFeature!.width)
        let cropScaleSIMD = SIMD2(imgW, imgH)

        let hScale = max(imgH, imgW) / imgH
        let wScale = max(imgH, imgW) / imgW
        let wShift = (wScale - 1)/2.0
        let hShift = (hScale - 1)/2.0

        let scaleSIMD = SIMD4(wScale, hScale, wScale, hScale)
        let shiftSIMD = SIMD4(wShift, hShift, wShift, hShift)

        previousCenters = []
        if previousCenters.count < maxPalms {
            let xInput = BlazePalmModelInput(image: imageFeature!)
            guard let points = try? self.model!.prediction(from: xInput) else {
                return []
            }
            let rPointsMLArray = points.featureValue(for: "2229")?.multiArrayValue
            let rPoints = rPointsMLArray?.dataPointer.bindMemory(to:
                                                                    SIMD4<Float32>.self,
                                                                 capacity:
                                                                    rPointsMLArray!.count/4)
            // 896 x 8 x 2 -> 2 bounding box + 6 keypoints
            let rArray = [SIMD4<Float32>](UnsafeBufferPointer(start: rPoints, count: rPointsMLArray!.count/4))

            let cMLArray = points.featureValue(for: "2145")?.multiArrayValue
            let cPointer = cMLArray?.dataPointer.bindMemory(to: Float32.self, capacity: cMLArray!.count)
            let cArray = [Float32](UnsafeBufferPointer(start: cPointer, count: cMLArray!.count))

            // Apply custom NMS
            var cIndices = cArray.enumerated().filter({ $0.element >= self.minConfidence }).map({ $0.offset })
            cIndices.sort(by: { cArray[$0] > cArray[$1] })

            while cIndices.count > 0 && previousCenters.count < maxPalms {
                var overlapRs = [SIMD4<Float32>]()
                var overlapCscore: Float32 = 0.0
                var nonOverlapI = [Int]()
                for index in 0..<cIndices.count {
                    let iiou = IOU(rArray[cIndices[0]], rArray[cIndices[index]])
                    if iiou >= 0.5 {
                        overlapRs.append(cArray[cIndices[index]]*rArray[cIndices[index]])
                        overlapCscore += cArray[cIndices[index]]
                    } else {
                        nonOverlapI.append(cIndices[index])
                    }
                }
                cIndices = nonOverlapI
                let bestR = scaleSIMD * (overlapRs.reduce(SIMD4<Float32>(repeating: 0), +) / overlapCscore) - shiftSIMD
                let averageCscore = overlapCscore / Float(overlapRs.count)
                if averageCscore < minConfidence {
                    continue
                }
                // 0 -> x left top, 3 -> y right bottom
                // we need topXY, dimensions
                var newR = bestR
                newR.highHalf = bestR.highHalf - bestR.lowHalf

                newR.lowHalf -= newR.highHalf*(dimScale-1)*0.5 // scale the dimensions by 1.5
                newR.highHalf *= dimScale
                previousCenters.append(newR)
            }
        }

        if previousCenters.count == 0 {
            return []
        }

        let palmInputs = BlazeLandmarksBatchInput(batch: previousCenters.count)

        var palms: [Palm] = []
        var imageDataSIMDs: [SIMD4<Float>] = []
        for bestR in previousCenters {
            let croppedHand = cropCGImage(imageFeature!,
                                          xStart: Int(imgW*bestR[0]),
                                          yStart: Int(imgH*bestR[1]),
                                          width: Int(imgW*bestR[2]),
                                          height: Int(imgH*bestR[3]))
            if croppedHand == nil {
                palmInputs.count -= 1
                continue
            }
            palmInputs.append(xFeature: BlazeLandmarksModelInput(image: croppedHand!))
            let cropH = Float(croppedHand!.height)
            let cropW = Float(croppedHand!.width)

            imageDataSIMDs.append(SIMD4(cropW, cropH, bestR[0], bestR[1]))
        }

        previousCenters = []

        guard let palmsStuff = try? self.landmarkModel?.predictions(fromBatch: palmInputs) else {
            return []
        }

        for index in 0..<palmsStuff.count {

            let cropHScale = max(imageDataSIMDs[index][0], imageDataSIMDs[index][1]) / imageDataSIMDs[index][1]
            let cropWScale = max(imageDataSIMDs[index][0], imageDataSIMDs[index][1]) / imageDataSIMDs[index][0]
            let scaleWB = (cropWScale - 1)/2.0
            let scaleHB = (cropHScale - 1)/2.0
            let shiftScaleSIMD = SIMD4(cropWScale, cropHScale, scaleWB, scaleHB)

            let palmData = palmsStuff.features(at: index)
            let confidence = UnsafeMutablePointer<Float32>(
                OpaquePointer(
                    (palmData.featureValue(for: "2317")?.multiArrayValue!.dataPointer)!
                )
            )[0]
//            let handedness = UnsafeMutablePointer<Float32>(
//                OpaquePointer(
//                    (palmData.featureValue(for: "2341")?.multiArrayValue!.dataPointer)!
//                )
//            )[0]

            let kMLArray = palmData.featureValue(for: "2371")?.multiArrayValue
            let kPointer = kMLArray?.dataPointer.bindMemory(to: Float32.self, capacity: kMLArray!.count)
            let kArray = [Float32](UnsafeBufferPointer(start: kPointer, count: kMLArray!.count))
            var landmarks = [SIMD4<Float>?](repeating: nil, count: 21)
            for jndex in 0..<21 {
                var landmarksXYZ = SIMD4(kArray[3*jndex], kArray[3*jndex+1], kArray[3*jndex+2], 0.0)
                // SIMD3 and 4 take the same amount of space
                landmarksXYZ.lowHalf = landmarksXYZ.lowHalf * shiftScaleSIMD.lowHalf - shiftScaleSIMD.highHalf
                // landmark from [0,1] in the cropped image space
                landmarksXYZ.lowHalf *= imageDataSIMDs[index].lowHalf / cropScaleSIMD
                landmarksXYZ.lowHalf += imageDataSIMDs[index].highHalf
                // landmark from [0,1] in the full image space
                landmarks[jndex] = landmarksXYZ
            }
            // these landmarks are in [0,1] in full image space imgWH
            let palmVector = landmarks[9]!.lowHalf - landmarks[0]!.lowHalf
            let palmLength = sqrt((palmVector * palmVector).sum()) * 2.6
            let handLength = palmLength * dimScale
            let boundingBox = SIMD4(
                landmarks[9]![0]-handLength/2.0,
                landmarks[9]![1]-handLength/2.0,
                handLength,
                handLength
            )
            previousCenters.append(boundingBox)
            let bbRect = CGRect(
                x: Double(boundingBox[0]),
                y: Double(boundingBox[1]),
                width: Double(palmLength),
                height: Double(palmLength)
            )

            palms.append(Palm(landmarks: landmarks, bBox: bbRect, conf: confidence))
        }

        return palms
    }
}
