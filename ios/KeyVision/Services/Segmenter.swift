import Accelerate
import CoreImage
import CoreVideo
import Foundation
import UIKit
import Vision

enum SegmentationError: Error, LocalizedError {
    case noForeground
    case badAspectRatio(Float)
    case tooSmall(Float)
    case tooLarge(Float)
    case tooBlurry(Float)
    case renderFailed

    var errorDescription: String? {
        switch self {
        case .noForeground: return "No foreground subject detected"
        case .badAspectRatio(let r): return "Aspect ratio \(String(format: "%.2f", r)) outside 1.5–8.0"
        case .tooSmall(let p): return "Key too small (\(String(format: "%.1f", p * 100))% of frame)"
        case .tooLarge(let p): return "Key too large (\(String(format: "%.1f", p * 100))% of frame)"
        case .tooBlurry(let v): return "Image too blurry (variance \(String(format: "%.1f", v)) < 100)"
        case .renderFailed: return "Failed to render masked image"
        }
    }
}

/// Segments a key from a CVPixelBuffer using Vision (iOS 17+) and returns a 224×224 CGImage.
final class Segmenter {
    static let shared = Segmenter()
    private init() {}

    private let canvasSize = 224
    private let aspectMin: Float = 1.5
    private let aspectMax: Float = 8.0
    private let areaMin: Float = 0.05
    private let areaMax: Float = 0.80
    private let blurThreshold: Float = 100.0

    @available(iOS 17.0, *)
    func segment(_ pixelBuffer: CVPixelBuffer) async throws -> CGImage {
        // Step 1: VNGenerateForegroundInstanceMaskRequest
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        let request = VNGenerateForegroundInstanceMaskRequest()
        try handler.perform([request])

        guard let observation = request.results?.first else {
            throw SegmentationError.noForeground
        }

        // Step 2: Apply mask → white background
        let maskedBuffer = try observation.generateMaskedImage(
            ofInstances: observation.allInstances,
            from: handler,
            croppedToInstancesExtent: false
        )

        // Step 3: Convert to CGImage for manipulation
        let ciImage = CIImage(cvPixelBuffer: maskedBuffer)
        let context = CIContext()
        guard let cgFull = context.createCGImage(ciImage, from: ciImage.extent) else {
            throw SegmentationError.renderFailed
        }

        // Set masked-out pixels to white by compositing over a white background
        let W = cgFull.width, H = cgFull.height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                                  bytesPerRow: W * 4, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw SegmentationError.renderFailed }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
        ctx.draw(cgFull, in: CGRect(x: 0, y: 0, width: W, height: H))
        guard let composite = ctx.makeImage() else { throw SegmentationError.renderFailed }

        // Step 4: Bounding box of non-white region
        let bbox = try boundingBoxOfForeground(image: composite)

        // Step 5: Area gate
        let imgArea = Float(W * H)
        let bboxArea = Float(bbox.width * bbox.height)
        let areaPct = bboxArea / imgArea
        if areaPct < areaMin { throw SegmentationError.tooSmall(areaPct) }
        if areaPct > areaMax { throw SegmentationError.tooLarge(areaPct) }

        // Step 6: Aspect ratio gate
        let bw = Float(bbox.width), bh = Float(bbox.height)
        let ratio = max(bw, bh) / min(bw, bh)
        if ratio < aspectMin || ratio > aspectMax { throw SegmentationError.badAspectRatio(ratio) }

        // Step 7: Crop with 10% padding, rotate so long axis is horizontal
        let padX = CGFloat(Int(bbox.width  * 0.10))
        let padY = CGFloat(Int(bbox.height * 0.10))
        let cropX = max(0,       bbox.origin.x - padX)
        let cropY = max(0,       bbox.origin.y - padY)
        let cropW = min(CGFloat(W) - cropX, bbox.width  + 2 * padX)
        let cropH = min(CGFloat(H) - cropY, bbox.height + 2 * padY)
        let cropRect = CGRect(x: cropX, y: cropY, width: cropW, height: cropH)
        guard var crop = composite.cropping(to: cropRect) else { throw SegmentationError.renderFailed }

        // Rotate so long axis is horizontal
        if crop.height > crop.width {
            crop = try rotate90(crop)
        }

        // Letterbox onto 224×224 white canvas
        let canvas = try letterbox(crop, to: canvasSize)

        // Step 8: Blur gate
        let blurScore = try laplacianVariance(canvas)
        if blurScore < blurThreshold { throw SegmentationError.tooBlurry(blurScore) }

        return canvas
    }

    // MARK: - Helpers

    private func boundingBoxOfForeground(image: CGImage) throws -> CGRect {
        let W = image.width, H = image.height
        guard let data = image.dataProvider?.data,
              let ptr = CFDataGetBytePtr(data) else { throw SegmentationError.renderFailed }
        let bpp = image.bitsPerPixel / 8

        var minX = W, maxX = 0, minY = H, maxY = 0
        for y in 0..<H {
            for x in 0..<W {
                let offset = (y * W + x) * bpp
                let r = Float(ptr[offset]) / 255.0
                let g = Float(ptr[offset + 1]) / 255.0
                let b = Float(ptr[offset + 2]) / 255.0
                // Non-white pixel → part of the key
                if r < 0.98 || g < 0.98 || b < 0.98 {
                    minX = min(minX, x); maxX = max(maxX, x)
                    minY = min(minY, y); maxY = max(maxY, y)
                }
            }
        }
        guard maxX > minX && maxY > minY else { throw SegmentationError.noForeground }
        return CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)
    }

    private func rotate90(_ image: CGImage) throws -> CGImage {
        let W = image.height, H = image.width
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8,
                                  bytesPerRow: W * 4, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw SegmentationError.renderFailed }
        ctx.translateBy(x: CGFloat(W), y: 0)
        ctx.rotate(by: .pi / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: CGFloat(image.width), height: CGFloat(image.height)))
        guard let rotated = ctx.makeImage() else { throw SegmentationError.renderFailed }
        return rotated
    }

    private func letterbox(_ image: CGImage, to size: Int) throws -> CGImage {
        let scale = min(Float(size) / Float(image.width), Float(size) / Float(image.height))
        let newW = Int(Float(image.width) * scale)
        let newH = Int(Float(image.height) * scale)
        let ox = (size - newW) / 2, oy = (size - newH) / 2

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: size * 4, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw SegmentationError.renderFailed }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.draw(image, in: CGRect(x: ox, y: oy, width: newW, height: newH))
        guard let result = ctx.makeImage() else { throw SegmentationError.renderFailed }
        return result
    }

    /// Laplacian variance via vImage — mirrors the server's blur gate.
    private func laplacianVariance(_ image: CGImage) throws -> Float {
        let size = image.width  // square
        guard let data = image.dataProvider?.data,
              let srcPtr = CFDataGetBytePtr(data) else { throw SegmentationError.renderFailed }
        let bpp = image.bitsPerPixel / 8

        // Convert to grayscale float
        var gray = [Float](repeating: 0, count: size * size)
        for i in 0..<size * size {
            let offset = i * bpp
            let r = Float(srcPtr[offset]), g = Float(srcPtr[offset + 1]), b = Float(srcPtr[offset + 2])
            gray[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        // 3×3 Laplacian kernel: [0,1,0, 1,-4,1, 0,1,0]
        var kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
        var lap = [Float](repeating: 0, count: size * size)

        var srcBuf: vImage_Buffer!
        var dstBuf: vImage_Buffer!
        var kPtr: UnsafeMutablePointer<Float>!

        gray.withUnsafeBufferPointer { grayBuf in
            srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: grayBuf.baseAddress!),
                                   height: vImagePixelCount(size), width: vImagePixelCount(size),
                                   rowBytes: size * MemoryLayout<Float>.size)
        }
        lap.withUnsafeMutableBufferPointer { lapBuf in
            dstBuf = vImage_Buffer(data: lapBuf.baseAddress!,
                                   height: vImagePixelCount(size), width: vImagePixelCount(size),
                                   rowBytes: size * MemoryLayout<Float>.size)
        }
        kernel.withUnsafeMutableBufferPointer { kBuf in
            kPtr = kBuf.baseAddress!
        }
        vImageConvolve_PlanarF(&srcBuf, &dstBuf, nil, 0, 0, kPtr, 3, 3, 0, vImage_Flags(kvImageEdgeExtend))

        // Compute variance
        var mean: Float = 0, variance: Float = 0
        vDSP_meanv(lap, 1, &mean, vDSP_Length(lap.count))
        var shifted = lap.map { $0 - mean }
        vDSP_measqv(&shifted, 1, &variance, vDSP_Length(shifted.count))
        return variance
    }
}
