import Accelerate
import CoreML
import Foundation
import UIKit

enum EmbedderError: Error, LocalizedError {
    case modelNotFound
    case pixelBufferFailed
    case predictionFailed
    case badNorm(Float)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:   return "DINOv2 model not found in bundle — run scripts/convert_dinov2_to_coreml.py first"
        case .pixelBufferFailed: return "Failed to render CGImage into a pixel buffer"
        case .predictionFailed:  return "CoreML prediction failed"
        case .badNorm(let n):    return "Embedding norm \(String(format: "%.4f", n)) not close to 1.0"
        }
    }
}

/// Wraps the CoreML DINOv2 model and returns a 768-d L2-normalized embedding.
/// Uses the generic MLModel API so it does not depend on Xcode's auto-generated
/// DINOv2 Swift class — the model is loaded by URL from the bundle.
final class Embedder {
    static let shared = Embedder()

    private let model: MLModel

    private init() {
        do {
            model = try Self.loadModel()
        } catch {
            fatalError("Embedder: \(error.localizedDescription)")
        }
    }

    // MARK: - Public

    func embed(_ image: CGImage) throws -> [Float] {
        let input = try makeInputArray(from: image)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: input)])
        let output = try model.prediction(from: provider)

        guard let name = output.featureNames.first,
              let arr  = output.featureValue(for: name)?.multiArrayValue else {
            throw EmbedderError.predictionFailed
        }

        var embedding = [Float](repeating: 0, count: arr.count)
        // Access raw float32 data directly via pointer — avoids NSNumber boxing overhead
        arr.withUnsafeBytes { ptr in
            let floatPtr = ptr.bindMemory(to: Float.self)
            for i in 0..<arr.count { embedding[i] = floatPtr[i] }
        }

        // Verify norm ≈ 1.0 (model bakes in L2 normalization)
        var norm: Float = 0
        vDSP_dotpr(embedding, 1, embedding, 1, &norm, vDSP_Length(embedding.count))
        norm = sqrt(norm)
        guard (0.99...1.01).contains(norm) else { throw EmbedderError.badNorm(norm) }

        return embedding
    }

    // MARK: - Model loading

    private static func loadModel() throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // Xcode compiles DINOv2.mlpackage → DINOv2.mlmodelc in the bundle.
        if let url = Bundle.main.url(forResource: "DINOv2", withExtension: "mlmodelc") {
            return try MLModel(contentsOf: url, configuration: config)
        }

        // Fallback: the .mlpackage was copied as-is (e.g. folder reference).
        // Compile it on first launch and cache the result.
        guard let pkgURL = Bundle.main.url(forResource: "DINOv2", withExtension: "mlpackage") else {
            throw EmbedderError.modelNotFound
        }

        let cacheURL = try compiledModelCacheURL()
        if FileManager.default.fileExists(atPath: cacheURL.path) {
            if let model = try? MLModel(contentsOf: cacheURL, configuration: config) {
                return model
            }
        }

        // Compile (takes a few seconds on first launch)
        let compiledURL = try MLModel.compileModel(at: pkgURL)
        try? FileManager.default.removeItem(at: cacheURL)
        try FileManager.default.copyItem(at: compiledURL, to: cacheURL)
        return try MLModel(contentsOf: cacheURL, configuration: config)
    }

    private static func compiledModelCacheURL() throws -> URL {
        try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask,
                                     appropriateFor: nil, create: true)
            .appendingPathComponent("DINOv2.mlmodelc")
    }

    // MARK: - Input preparation

    /// Converts a 224×224 CGImage to an MLMultiArray of shape (1, 3, 224, 224)
    /// with float32 values in [0, 1]. The model applies ImageNet normalization internally.
    private func makeInputArray(from image: CGImage) throws -> MLMultiArray {
        let size = 224

        // Render into a flat RGBA byte buffer
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: size, height: size,
                                  bitsPerComponent: 8, bytesPerRow: size * 4,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw EmbedderError.pixelBufferFailed }

        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        guard let pixelData = ctx.data else { throw EmbedderError.pixelBufferFailed }

        let multiArray = try MLMultiArray(shape: [1, 3, size, size] as [NSNumber], dataType: .float32)

        // Write directly into the MLMultiArray's float32 buffer (NCHW layout)
        multiArray.withUnsafeMutableBytes { dst, strides in
            let floatDst = dst.bindMemory(to: Float.self)
            let pixelSrc = pixelData.bindMemory(to: UInt8.self, capacity: size * size * 4)
            let n = size * size
            for i in 0..<n {
                let base = i * 4
                floatDst[i]         = Float(pixelSrc[base])     / 255.0  // R
                floatDst[n + i]     = Float(pixelSrc[base + 1]) / 255.0  // G
                floatDst[2 * n + i] = Float(pixelSrc[base + 2]) / 255.0  // B
            }
        }

        return multiArray
    }
}
