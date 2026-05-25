import SwiftUI

struct SettingsView: View {
    @AppStorage("serverURL") private var serverURL = "http://localhost:8000"
    @StateObject private var sync = SyncService.shared

    private var lastSyncText: String {
        guard let date = sync.lastSyncDate else { return "Never" }
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("Server URL", text: $serverURL)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                }

                Section("Sync") {
                    LabeledContent("Last synced", value: lastSyncText)

                    Button {
                        Task { await SyncService.shared.syncAll() }
                    } label: {
                        HStack {
                            Text("Sync Now")
                            Spacer()
                            if sync.isSyncing {
                                ProgressView().scaleEffect(0.8)
                            }
                        }
                    }
                    .disabled(sync.isSyncing)
                }

                if let error = sync.errorMessage {
                    Section("Error") {
                        Text(error).foregroundColor(.red).font(.caption)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}
