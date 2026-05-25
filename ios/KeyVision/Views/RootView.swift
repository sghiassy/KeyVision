import SwiftUI

struct RootView: View {
    var body: some View {
        TabView {
            CameraTabView()
                .tabItem { Label("Camera", systemImage: "camera.fill") }

            KeysListView()
                .tabItem { Label("Keys", systemImage: "key.fill") }

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape.fill") }
        }
    }
}
