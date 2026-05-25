import SwiftUI

struct CameraTabView: View {
    @StateObject private var controller = CameraController()

    var body: some View {
        NavigationStack {
            ZStack {
                if controller.isAuthorized {
                    CameraPreviewView(session: controller.session)
                        .ignoresSafeArea()
                } else {
                    Color.black.ignoresSafeArea()
                    VStack(spacing: 16) {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 48))
                            .foregroundColor(.white.opacity(0.5))
                        Text("Camera access required")
                            .foregroundColor(.white)
                        Button("Grant Access") {
                            Task { await controller.requestAccess() }
                        }
                        .buttonStyle(.bordered)
                    }
                }

                RecognitionOverlay(result: controller.latestResult)
            }
        }
        .task {
            await controller.requestAccess()
            controller.startSession()
        }
        .onDisappear { controller.stopSession() }
    }
}
