import SwiftUI

struct KeysListView: View {
    @State private var keys: [Key] = []
    @State private var showEnrollment = false

    var body: some View {
        NavigationStack {
            List {
                ForEach(keys) { key in
                    NavigationLink(destination: KeyDetailView(keyId: key.id, label: key.label)) {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(key.label).font(.headline)
                                Text("\(key.imageCount) image\(key.imageCount == 1 ? "" : "s")")
                                    .font(.caption).foregroundColor(.secondary)
                            }
                            Spacer()
                            syncBadge(key.syncStatus)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Keys")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button { showEnrollment = true } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showEnrollment, onDismiss: { refresh() }) {
                EnrollmentFlowView()
            }
            .onAppear { refresh() }
        }
    }

    private func refresh() {
        keys = LocalStore.shared.allKeys()
    }

    @ViewBuilder
    private func syncBadge(_ status: Key.SyncStatus) -> some View {
        switch status {
        case .synced: EmptyView()
        case .pending:
            Image(systemName: "arrow.triangle.2.circlepath")
                .foregroundColor(.orange).font(.caption)
        case .failed:
            Image(systemName: "exclamationmark.triangle")
                .foregroundColor(.red).font(.caption)
        }
    }
}
