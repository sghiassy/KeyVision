import AVFoundation
import CoreImage
import SwiftUI
import UIKit

/// Single-step capture: shows live camera, lets user snap, preview, confirm or retake.
struct CaptureStepView: View {
    let prompt: String
    let onConfirm: (UIImage) -> Void

    @State private var capturedImage: UIImage? = nil
    @State private var showPicker = false

    var body: some View {
        VStack(spacing: 24) {
            Text(prompt)
                .font(.title3).bold()
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            if let captured = capturedImage {
                Image(uiImage: captured)
                    .resizable().scaledToFit()
                    .frame(maxHeight: 300)
                    .cornerRadius(12)

                HStack(spacing: 20) {
                    Button("Retake") {
                        capturedImage = nil
                        showPicker = true
                    }
                    .buttonStyle(.bordered)

                    Button("Use This Photo") {
                        onConfirm(captured)
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.secondary.opacity(0.15))
                    .frame(height: 300)
                    .overlay(
                        VStack(spacing: 12) {
                            Image(systemName: "camera.circle")
                                .font(.system(size: 48))
                                .foregroundColor(.secondary)
                            Text("Tap to capture")
                                .foregroundColor(.secondary)
                        }
                    )
                    .onTapGesture { showPicker = true }
            }

            Spacer()
        }
        .padding()
        .sheet(isPresented: $showPicker) {
            ImageCaptureSheet { image in
                capturedImage = image
            }
        }
        .onAppear { showPicker = true }
    }
}

/// UIImagePickerController wrapped for SwiftUI.
private struct ImageCaptureSheet: UIViewControllerRepresentable {
    let onCapture: (UIImage) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onCapture: onCapture) }

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = UIImagePickerController.isSourceTypeAvailable(.camera) ? .camera : .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onCapture: (UIImage) -> Void
        init(onCapture: @escaping (UIImage) -> Void) { self.onCapture = onCapture }

        func imagePickerController(_ picker: UIImagePickerController,
                                    didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            picker.dismiss(animated: true)
            if let img = info[.originalImage] as? UIImage { onCapture(img) }
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}
