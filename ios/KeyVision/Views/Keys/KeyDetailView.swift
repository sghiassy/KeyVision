import SwiftUI

struct KeyDetailView: View {
    let keyId: String
    let label: String

    @State private var imageIds: [String] = []
    @Environment(\.dismiss) private var dismiss

    private let columns = [GridItem(.adaptive(minimum: 100))]

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(imageIds, id: \.self) { imageId in
                    AsyncImage(url: thumbnailURL(imageId)) { phase in
                        switch phase {
                        case .success(let img):
                            img.resizable().aspectRatio(contentMode: .fill)
                                .frame(width: 100, height: 100).clipped()
                                .cornerRadius(8)
                        default:
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.secondary.opacity(0.2))
                                .frame(width: 100, height: 100)
                        }
                    }
                }
            }
            .padding()
        }
        .navigationTitle(label)
        .toolbar {
            ToolbarItem(placement: .destructiveAction) {
                Button("Delete Key", role: .destructive) { deleteKey() }
            }
        }
        .onAppear { loadImages() }
    }

    private func loadImages() {
        // Images are identified by the embeddings stored locally
        let embeddings = LocalStore.shared.allEmbeddings()
        imageIds = embeddings.filter { $0.keyId == keyId }.map(\.imageId)
    }

    private func deleteKey() {
        LocalStore.shared.deleteKey(keyId: keyId)
        Task {
            // Best-effort server delete
            if let url = URL(string: "\(UserDefaults.standard.string(forKey: "serverURL") ?? "http://localhost:8000")/keys/\(keyId)") {
                var req = URLRequest(url: url)
                req.httpMethod = "DELETE"
                try? await URLSession.shared.data(for: req)
            }
        }
        dismiss()
    }

    private func thumbnailURL(_ imageId: String) -> URL? {
        let base = UserDefaults.standard.string(forKey: "serverURL") ?? "http://localhost:8000"
        return URL(string: "\(base)/keys/\(keyId)/images/\(imageId)")
    }
}
