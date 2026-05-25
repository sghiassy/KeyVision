import SwiftUI

@main
struct KeyVisionApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
                .task {
                    // Sync from server on launch
                    await SyncService.shared.syncAll()
                }
        }
    }
}
