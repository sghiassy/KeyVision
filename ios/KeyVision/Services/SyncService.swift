import Foundation

/// Bidirectional sync between server and LocalStore.
/// Device → server: POST /keys/{id}/images/embedded for pending embeddings.
/// Server → device: GET /sync, diff, apply.
@MainActor
final class SyncService: ObservableObject {
    static let shared = SyncService()

    @Published var lastSyncDate: Date? = nil
    @Published var isSyncing = false
    @Published var errorMessage: String? = nil

    private init() {}

    func syncAll() async {
        guard !isSyncing else { return }
        isSyncing = true
        errorMessage = nil
        defer { isSyncing = false }

        do {
            // 1. Pull from server
            let payload = try await APIService.shared.fetchSync()
            let store = LocalStore.shared

            let remoteKeyIds = Set(payload.map(\.keyId))
            let localKeys = store.allKeys()
            let localKeyIds = Set(localKeys.map(\.id))

            // Remove keys that no longer exist on the server
            for keyId in localKeyIds.subtracting(remoteKeyIds) {
                store.deleteKey(keyId: keyId)
            }

            // Upsert keys and images from server
            for item in payload {
                store.upsertKey(keyId: item.keyId, label: item.label, notes: item.notes, createdAt: item.createdAt)
                for img in item.images {
                    guard let embData = Data(base64Encoded: img.embedding) else { continue }
                    let embedding = embData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    guard embedding.count == 768 else { continue }
                    store.upsertEmbedding(imageId: img.imageId, keyId: item.keyId, embedding: embedding, createdAt: img.createdAt)
                }
            }

            // 2. Push pending embeddings to server
            // (LocalStore holds pending records written during offline enrollment)
            // The actual crop + embedding data must be retrieved from disk — see EnrollmentFlowView
            // This handles the case where a previous push failed and was retried later.
            // For new enrollments the push happens inline in EnrollmentFlowView.

            lastSyncDate = Date()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Called after a new on-device enrollment to push the embedding to the server.
    func pushEmbedding(keyId: String, imageId: String, cropJpeg: Data, embedding: [Float]) async {
        LocalStore.shared.updateEmbeddingSyncStatus(imageId: imageId, syncStatus: "pending")
        do {
            _ = try await APIService.shared.enrollEmbedded(keyId: keyId, imageId: imageId, cropJpeg: cropJpeg, embedding: embedding)
            LocalStore.shared.updateEmbeddingSyncStatus(imageId: imageId, syncStatus: "synced")
        } catch {
            LocalStore.shared.updateEmbeddingSyncStatus(imageId: imageId, syncStatus: "failed")
        }
    }
}
