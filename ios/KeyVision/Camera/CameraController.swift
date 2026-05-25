import AVFoundation
import Combine
import CoreVideo
import Foundation
import UIKit

/// Manages AVCaptureSession and drives the real-time recognition loop.
@MainActor
final class CameraController: NSObject, ObservableObject {
    @Published var latestResult: MatchResult? = nil
    @Published var isAuthorized = false

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let outputQueue = DispatchQueue(label: "com.keyvision.camera.output")

    private var frameCount = 0
    private let frameInterval = 15  // process every 15th frame ≈ 2 fps at 30fps
    private var recognitionTask: Task<Void, Never>? = nil
    private var resultHideTask: Task<Void, Never>? = nil

    var previewLayer: AVCaptureVideoPreviewLayer {
        AVCaptureVideoPreviewLayer(session: session)
    }

    func requestAccess() async {
        let status = await AVCaptureDevice.requestAccess(for: .video)
        isAuthorized = status
        if status { configureSession() }
    }

    func startSession() {
        guard isAuthorized else { return }
        Task.detached(priority: .userInitiated) { [weak self] in
            self?.session.startRunning()
        }
    }

    func stopSession() {
        Task.detached { [weak self] in
            self?.session.stopRunning()
        }
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
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                    didOutput sampleBuffer: CMSampleBuffer,
                                    from connection: AVCaptureConnection) {
        frameCount += 1
        guard frameCount % frameInterval == 0 else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Ignore if a recognition is already in flight
        guard recognitionTask == nil || recognitionTask!.isCancelled else { return }

        recognitionTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            if #available(iOS 17.0, *) {
                let embeddings = LocalStore.shared.allEmbeddings()
                guard !embeddings.isEmpty else { return }
                do {
                    let results = try await RecognitionEngine.shared.recognize(pixelBuffer: pixelBuffer, embeddings: embeddings)
                    let top = results.first
                    await MainActor.run {
                        if top?.confidence == .high || top?.confidence == .maybe {
                            self.latestResult = top
                            self.scheduleResultHide()
                        }
                        self.recognitionTask = nil
                    }
                } catch {
                    // Segmentation failures are silent during live feed
                    await MainActor.run { self.recognitionTask = nil }
                }
            }
        }
    }

    private func scheduleResultHide() {
        resultHideTask?.cancel()
        resultHideTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_000_000_000)  // 2s
            self.latestResult = nil
        }
    }
}
