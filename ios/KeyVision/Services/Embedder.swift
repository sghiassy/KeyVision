import Accelerate
import CoreML
import CoreVideo
import Foundation
import UIKit

enum EmbedderError: Error, LocalizedError {
    case pixelBufferFailed
    case predictionFailed
    case badNorm(Float)

    var errorDescription: String? {
        switch self {
        case .pixelBufferFailed: return "Failed to create CVPixelBuffer from image"
        case .predictionFailed: return "CoreML prediction failed"
        case .badNorm(let n): return "Embedding norm \(String(format: "%.4f", n)) not close to 1.0"
        }
    }
}

/// Wraps the CoreML DINOv2 model and returns a 768-d L2-normalized embedding.
final class Embedder {
    static let shared = Embedder()

    private let model: DINOv2

    private init() {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        // swiftlint:disable force_try
        model = try! DINOv2(configuration: config)
        // swiftlint:enable force_try
    }

    func embed(_ image: CGImage) throws -> [Float] {
        guard let pixelBuffer = pixelBuffer(from: image) else {
            throw EmbedderError.pixelBufferFailed
        }

        guard let output = try? model.prediction(x: pixelBuffer) else {
            throw EmbedderError.predictionFailed
        }

        let multiArray = output.featureValue(for: "var_\(768)")?.multiArrayValue
            ?? output.featureValue(for: "output")?.multiArrayValue
            ?? firstMultiArray(in: output)

        guard let arr = multiArray else { throw EmbedderError.predictionFailed }

        var embedding = [Float](repeating: 0, count: arr.count)
        for i in 0..<arr.count {
            embedding[i] = arr[i].floatValue
        }

        // Verify norm
        var norm: Float = 0
        vDSP_dotpr(embedding, 1, embedding, 1, &norm, vDSP_Length(embedding.count))
        norm = sqrt(norm)
        guard (0.99...1.01).contains(norm) else { throw EmbedderError.badNorm(norm) }

        return embedding
    }

    private func pixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        let size = 224
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, size, size,
                                   kCVPixelFormatType_32BGRA,
                                   attrs as CFDictionary, &pixelBuffer) == kCVReturnSuccess,
              let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: base, width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }

        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return buffer
    }

    private func firstMultiArray(in provider: MLFeatureProvider) -> MLMultiArray? {
        for name in provider.featureNames {
            if let val = provider.featureValue(for: name)?.multiArrayValue { return val }
        }
        return nil
    }
}
