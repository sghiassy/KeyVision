import Accelerate
import CoreVideo
import Foundation

enum Confidence: String {
    case high, maybe, noMatch = "no_match"

    static func classify(_ similarity: Float) -> Confidence {
        if similarity >= 0.85 { return .high }
        if similarity >= 0.65 { return .maybe }
        return .noMatch
    }
}

struct MatchResult {
    let keyId: String
    let label: String
    let similarity: Float
    let confidence: Confidence
}

/// Pure recognition pipeline: segment → embed → match. No UI dependencies.
final class RecognitionEngine {
    static let shared = RecognitionEngine()

    private let segmenter = Segmenter.shared
    private let embedder = Embedder.shared

    private init() {}

    @available(iOS 17.0, *)
    func recognize(pixelBuffer: CVPixelBuffer, embeddings: [EmbeddingEntry]) async throws -> [MatchResult] {
        let crop = try await segmenter.segment(pixelBuffer)
        let queryEmb = try embedder.embed(crop)
        return match(query: queryEmb, stored: embeddings)
    }

    func match(query: [Float], stored: [EmbeddingEntry]) -> [MatchResult] {
        guard !stored.isEmpty else { return [] }

        // Group embeddings by keyId, take max similarity per key
        var bestByKey: [String: (label: String, similarity: Float)] = [:]

        for entry in stored {
            let sim = dotProduct(query, entry.embedding)
            if let current = bestByKey[entry.keyId] {
                if sim > current.similarity {
                    bestByKey[entry.keyId] = (entry.label, sim)
                }
            } else {
                bestByKey[entry.keyId] = (entry.label, sim)
            }
        }

        let results = bestByKey.map { keyId, info in
            MatchResult(keyId: keyId, label: info.label, similarity: info.similarity,
                        confidence: Confidence.classify(info.similarity))
        }
        return results.sorted { $0.similarity > $1.similarity }
    }

    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
}
