import Foundation

struct EmbeddingEntry {
    let imageId: String
    let keyId: String
    let label: String
    let embedding: [Float]  // 768-d, L2-normalized
}
