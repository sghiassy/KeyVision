import Foundation
import SQLite3

/// Thread-safe on-device SQLite store for keys and embeddings.
final class LocalStore {
    static let shared = LocalStore()

    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "com.keyvision.localstore", attributes: .concurrent)

    private init() {
        let url = Self.dbURL
        if sqlite3_open(url.path, &db) != SQLITE_OK {
            fatalError("Cannot open SQLite database at \(url.path)")
        }
        migrate()
    }

    deinit { sqlite3_close(db) }

    private static var dbURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("keyvision.db")
    }

    private func migrate() {
        let ddl = """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS keys (
                key_id     TEXT PRIMARY KEY,
                label      TEXT NOT NULL,
                notes      TEXT,
                created_at TEXT NOT NULL,
                sync_status TEXT NOT NULL DEFAULT 'synced'
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                image_id   TEXT PRIMARY KEY,
                key_id     TEXT NOT NULL REFERENCES keys(key_id) ON DELETE CASCADE,
                embedding  BLOB NOT NULL,
                created_at TEXT NOT NULL,
                sync_status TEXT NOT NULL DEFAULT 'synced'
            );
        """
        sqlite3_exec(db, ddl, nil, nil, nil)
    }

    // MARK: - Keys

    func upsertKey(keyId: String, label: String, notes: String?, createdAt: String, syncStatus: String = "synced") {
        queue.async(flags: .barrier) {
            let sql = """
                INSERT INTO keys (key_id, label, notes, created_at, sync_status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET label=excluded.label, notes=excluded.notes, sync_status=excluded.sync_status
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(self.db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, keyId, -1, Self.transient)
            sqlite3_bind_text(stmt, 2, label, -1, Self.transient)
            if let notes { sqlite3_bind_text(stmt, 3, notes, -1, Self.transient) } else { sqlite3_bind_null(stmt, 3) }
            sqlite3_bind_text(stmt, 4, createdAt, -1, Self.transient)
            sqlite3_bind_text(stmt, 5, syncStatus, -1, Self.transient)
            sqlite3_step(stmt)
        }
    }

    func deleteKey(keyId: String) {
        queue.async(flags: .barrier) {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(self.db, "DELETE FROM keys WHERE key_id = ?", -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, keyId, -1, Self.transient)
            sqlite3_step(stmt)
        }
    }

    func allKeys() -> [Key] {
        var result: [Key] = []
        queue.sync {
            let sql = """
                SELECT k.key_id, k.label, k.notes, k.created_at, k.sync_status,
                       COUNT(e.image_id) AS image_count
                FROM keys k LEFT JOIN embeddings e ON k.key_id = e.key_id
                GROUP BY k.key_id ORDER BY k.created_at
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(self.db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                let keyId = String(cString: sqlite3_column_text(stmt, 0))
                let label = String(cString: sqlite3_column_text(stmt, 1))
                let notes = sqlite3_column_type(stmt, 2) != SQLITE_NULL ? String(cString: sqlite3_column_text(stmt, 2)) : nil
                let createdAt = String(cString: sqlite3_column_text(stmt, 3))
                let syncStr = String(cString: sqlite3_column_text(stmt, 4))
                let imageCount = Int(sqlite3_column_int(stmt, 5))
                let sync = Key.SyncStatus(rawValue: syncStr) ?? .synced
                result.append(Key(id: keyId, label: label, notes: notes, createdAt: createdAt, imageCount: imageCount, syncStatus: sync))
            }
        }
        return result
    }

    // MARK: - Embeddings

    func upsertEmbedding(imageId: String, keyId: String, embedding: [Float], createdAt: String, syncStatus: String = "synced") {
        queue.async(flags: .barrier) {
            let sql = """
                INSERT INTO embeddings (image_id, key_id, embedding, created_at, sync_status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO NOTHING
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(self.db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            let blob = embedding.withUnsafeBytes { Data($0) }
            sqlite3_bind_text(stmt, 1, imageId, -1, Self.transient)
            sqlite3_bind_text(stmt, 2, keyId, -1, Self.transient)
            blob.withUnsafeBytes { ptr in
                sqlite3_bind_blob(stmt, 3, ptr.baseAddress, Int32(blob.count), Self.transient)
            }
            sqlite3_bind_text(stmt, 4, createdAt, -1, Self.transient)
            sqlite3_bind_text(stmt, 5, syncStatus, -1, Self.transient)
            sqlite3_step(stmt)
        }
    }

    func allEmbeddings() -> [EmbeddingEntry] {
        var result: [EmbeddingEntry] = []
        queue.sync {
            let sql = """
                SELECT e.image_id, e.key_id, e.embedding, k.label
                FROM embeddings e JOIN keys k ON e.key_id = k.key_id
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(self.db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                let imageId = String(cString: sqlite3_column_text(stmt, 0))
                let keyId = String(cString: sqlite3_column_text(stmt, 1))
                let blobLen = sqlite3_column_bytes(stmt, 2)
                let blobPtr = sqlite3_column_blob(stmt, 2)!
                let data = Data(bytes: blobPtr, count: Int(blobLen))
                let embedding = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                let label = String(cString: sqlite3_column_text(stmt, 3))
                result.append(EmbeddingEntry(imageId: imageId, keyId: keyId, label: label, embedding: embedding))
            }
        }
        return result
    }

    func updateEmbeddingSyncStatus(imageId: String, syncStatus: String) {
        queue.async(flags: .barrier) {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(self.db, "UPDATE embeddings SET sync_status = ? WHERE image_id = ?", -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            sqlite3_bind_text(stmt, 1, syncStatus, -1, Self.transient)
            sqlite3_bind_text(stmt, 2, imageId, -1, Self.transient)
            sqlite3_step(stmt)
        }
    }

    func pendingEmbeddings() -> [(imageId: String, keyId: String)] {
        var result: [(String, String)] = []
        queue.sync {
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(self.db, "SELECT image_id, key_id FROM embeddings WHERE sync_status = 'pending'", -1, &stmt, nil) == SQLITE_OK else { return }
            defer { sqlite3_finalize(stmt) }
            while sqlite3_step(stmt) == SQLITE_ROW {
                result.append((String(cString: sqlite3_column_text(stmt, 0)), String(cString: sqlite3_column_text(stmt, 1))))
            }
        }
        return result
    }

    // SQLite transient destructor constant
    private static let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
}
