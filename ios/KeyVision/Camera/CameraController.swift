import AVFoundation
import Combine
import CoreVideo
import Foundation
import UIKit

/// Manages AVCaptureSession and drives the real-time recognition loop.
///
/// NOT @MainActor — captureOutput fires on outputQueue (background thread).
/// @Published mutations are dispatched to the main actor explicitly.
final class CameraController: NSObject, ObservableObject {
    @Published var latestResult: MatchResult? = nil
    @Published var isAuthorized = false

    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let outputQueue = DispatchQueue(label: "com.keyvision.camera.output")

    // Accessed only from outputQueue — no actor isolation needed.
    private var frameCount = 0
    private let frameInterval = 15  // process every 15th frame ≈ 2 fps at 30fps
    private var recognitionInFlight = false

    func requestAccess() async {
        let status = await AVCaptureDevice.requestAccess(for: .video)
        await MainActor.run {
            isAuthorized = status
        }
        if status { configureSession() }
    }

    func startSession() {
        guard isAuthorized else { return }
        // AVCaptureSession.startRunning() must not be called on the main thread.
        outputQueue.async { self.session.startRunning() }
    }

    func stopSession() {
        outputQueue.async { self.session.stopRunning() }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .photo

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        session.commitConfiguration()
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    // Called on outputQueue — no actor isolation; all CameraController properties accessed
    // here must not be @MainActor. UI updates hop to MainActor explicitly.
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        frameCount += 1
        guard frameCount % frameInterval == 0 else { return }
        guard !recognitionInFlight else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        recognitionInFlight = true

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            if #available(iOS 17.0, *) {
                let embeddings = LocalStore.shared.allEmbeddings()
                guard !embeddings.isEmpty else {
                    self.recognitionInFlight = false
                    return
                }
                do {
                    let results = try await RecognitionEngine.shared.recognize(
                        pixelBuffer: pixelBuffer,
                        embeddings: embeddings
                    )
                    let top = results.first
                    self.recognitionInFlight = false

                    await MainActor.run {
                        if top?.confidence == .high || top?.confidence == .maybe {
                            self.latestResult = top
                            self.scheduleResultHide()
                        }
                    }
                } catch {
                    // Segmentation failures during live feed are silent.
                    self.recognitionInFlight = false
                }
            } else {
                self.recognitionInFlight = false
            }
        }
    }

    @MainActor
    private func scheduleResultHide() {
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            self.latestResult = nil
        }
    }
}
