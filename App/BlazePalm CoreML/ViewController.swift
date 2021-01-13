//
//  ViewController.swift
//  BlazePalm CoreML
//
//  Created by Vidur Satija on 07/09/20.
//  Copyright © 2020 Vidur Satija. All rights reserved.
//

import AVFoundation
import UIKit
import Vision

class PalmDetectionViewController: UIViewController {
    @IBOutlet var palmView: PalmOverlayView!

    let session = AVCaptureSession()
    var previewLayer: AVCaptureVideoPreviewLayer!

    let dataOutputQueue = DispatchQueue(
        label: "video",
        qos: .userInitiated,
        attributes: [],
        autoreleaseFrequency: .workItem)

    let bfm = BlazePalmModel()

    override func viewDidLoad() {
        super.viewDidLoad()
        configureCaptureSession()
        session.startRunning()
    }
}

// MARK: - Video Stuff

extension PalmDetectionViewController {
    func configureCaptureSession() {
    // Define the capture device we want to use
        session.sessionPreset = .hd1280x720
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .front) else {
                                                    fatalError("No front video camera available")
        }

        // Connect the camera to the capture session input
        do {
            let cameraInput = try AVCaptureDeviceInput(device: camera)
            session.addInput(cameraInput)
        } catch {
            fatalError(error.localizedDescription)
        }

        // Create the video data output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: dataOutputQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

        // Add the video output to the capture session
        session.addOutput(videoOutput)

        let videoConnection = videoOutput.connection(with: .video)
        videoConnection?.videoOrientation = .portrait
        videoConnection?.isVideoMirrored = true

        // Configure the preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.insertSublayer(previewLayer, at: 0)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate methods

extension PalmDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let palms = bfm.predict(for: imageBuffer)

        DispatchQueue.main.async {
            self.palmView.landmarks = palms
            self.palmView.setNeedsDisplay()
        }
    }
}
