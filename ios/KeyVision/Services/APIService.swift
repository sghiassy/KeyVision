import Foundation
import UIKit

/// URLSession wrappers for all KeyVision backend endpoints.
final class APIService {
    static let shared = APIService()

    private init() {}

    private var baseURL: String {
        UserDefaults.standard.string(forKey: "serverURL") ?? "http://localhost:8000"
    }

    // MARK: - Sync

    struct SyncPayload: Decodable {
        let keyId: String
        let label: String
        let notes: String?
        let createdAt: String
        let images: [SyncImage]

        enum CodingKeys: String, CodingKey {
            case keyId = "key_id", label, notes
            case createdAt = "created_at", images
        }
    }

    struct SyncImage: Decodable {
        let imageId: String
        let embedding: String  // base64 float32[768]
        let createdAt: String

        enum CodingKeys: String, CodingKey {
            case imageId = "image_id", embedding
            case createdAt = "created_at"
        }
    }

    func fetchSync() async throws -> [SyncPayload] {
        let url = URL(string: "\(baseURL)/sync")!
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode([SyncPayload].self, from: data)
    }

    // MARK: - Keys

    struct CreateKeyResponse: Decodable {
        let keyId: String
        enum CodingKeys: String, CodingKey { case keyId = "key_id" }
    }

    func createKey(label: String, notes: String?) async throws -> String {
        var request = URLRequest(url: URL(string: "\(baseURL)/keys")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = ["label": label]
        if let notes { body["notes"] = notes }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(CreateKeyResponse.self, from: data).keyId
    }

    // MARK: - Enroll embedded

    struct EnrollResponse: Decodable {
        let imageId: String
        let segmentationOk: Bool
        enum CodingKeys: String, CodingKey { case imageId = "image_id"; case segmentationOk = "segmentation_ok" }
    }

    /// Upload a pre-computed embedding + segmented crop to the backend.
    func enrollEmbedded(keyId: String, imageId: String, cropJpeg: Data, embedding: [Float]) async throws -> EnrollResponse {
        let url = URL(string: "\(baseURL)/keys/\(keyId)/images/embedded")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        func append(_ string: String) { body.append(Data(string.utf8)) }

        // image_id field
        append("--\(boundary)\r\nContent-Disposition: form-data; name=\"image_id\"\r\n\r\n\(imageId)\r\n")

        // image file
        append("--\(boundary)\r\nContent-Disposition: form-data; name=\"image\"; filename=\"crop.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n")
        body.append(cropJpeg)
        append("\r\n")

        // embedding blob (768 × float32 little-endian)
        let embBlob = embedding.withUnsafeBytes { Data($0) }
        append("--\(boundary)\r\nContent-Disposition: form-data; name=\"embedding\"; filename=\"embedding.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n")
        body.append(embBlob)
        append("\r\n")

        append("--\(boundary)--\r\n")
        request.httpBody = body

        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(EnrollResponse.self, from: data)
    }
}
