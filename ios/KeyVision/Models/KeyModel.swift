import Foundation

struct Key: Identifiable, Codable, Equatable {
    let id: String
    var label: String
    var notes: String?
    let createdAt: String
    var imageCount: Int
    var syncStatus: SyncStatus

    enum CodingKeys: String, CodingKey {
        case id = "key_id"
        case label, notes
        case createdAt = "created_at"
        case imageCount = "image_count"
        case syncStatus = "sync_status"
    }

    enum SyncStatus: String, Codable {
        case synced, pending, failed
    }
}
