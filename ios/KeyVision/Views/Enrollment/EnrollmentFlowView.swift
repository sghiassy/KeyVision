import SwiftUI
import UIKit

private let capturePrompts = [
    "Hold the key flat, front side up",
    "Flip the key over, back side up",
    "Hold the key at a slight angle",
]

/// Guided 3-step enrollment flow.
struct EnrollmentFlowView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var keyLabel = ""
    @State private var keyNotes = ""
    @State private var step = 0  // 0 = name entry, 1–3 = capture steps, 4 = saving
    @State private var capturedImages: [UIImage] = []
    @State private var errorMessage: String? = nil
    @State private var isSaving = false

    var body: some View {
        NavigationStack {
            Group {
                if step == 0 {
                    nameEntryView
                } else if step >= 1 && step <= 3 {
                    CaptureStepView(
                        prompt: capturePrompts[step - 1],
                        onConfirm: { image in
                            capturedImages.append(image)
                            if step < 3 { step += 1 } else { step = 4 }
                        }
                    )
                    .navigationTitle("Photo \(step) of 3")
                } else {
                    savingView
                }
            }
            .navigationTitle(step == 0 ? "New Key" : "")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .alert("Error", isPresented: .constant(errorMessage != nil), actions: {
                Button("OK") { errorMessage = nil }
            }, message: {
                Text(errorMessage ?? "")
            })
        }
        .onChange(of: step) { _, newStep in
            if newStep == 4 { Task { await saveKey() } }
        }
    }

    private var nameEntryView: some View {
        Form {
            Section("Key Name") {
                TextField("e.g. Front Door", text: $keyLabel)
            }
            Section("Notes (optional)") {
                TextField("e.g. deadbolt", text: $keyNotes)
            }
            Section {
                Button("Start Photo Capture") {
                    guard !keyLabel.trimmingCharacters(in: .whitespaces).isEmpty else {
                        errorMessage = "Please enter a name for the key."
                        return
                    }
                    step = 1
                }
                .disabled(keyLabel.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
    }

    private var savingView: some View {
        VStack(spacing: 20) {
            if isSaving {
                ProgressView("Saving key…")
            } else {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 64))
                    .foregroundColor(.green)
                Text("Key saved!")
                    .font(.title2)
                Button("Done") { dismiss() }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func saveKey() async {
        isSaving = true
        defer { isSaving = false }

        let label = keyLabel.trimmingCharacters(in: .whitespaces)
        let notes = keyNotes.trimmingCharacters(in: .whitespaces).isEmpty ? nil : keyNotes.trimmingCharacters(in: .whitespaces)

        do {
            // Create key on server and get a shared key_id
            let keyId = try await APIService.shared.createKey(label: label, notes: notes)

            // Store key locally
            let now = ISO8601DateFormatter().string(from: Date())
            LocalStore.shared.upsertKey(keyId: keyId, label: label, notes: notes, createdAt: now, syncStatus: "synced")

            // Process and store each captured image
            for image in capturedImages {
                guard let cgImage = image.cgImage else { continue }
                let imageId = UUID().uuidString

                // Segment + embed
                guard #available(iOS 17.0, *) else { continue }
                let crop: CGImage
                let embedding: [Float]
                do {
                    crop = try await Segmenter.shared.segment(cgImageToPixelBuffer(cgImage)!)
                    embedding = try Embedder.shared.embed(crop)
                } catch {
                    // Skip images that fail segmentation (e.g., blurry) but continue with others
                    continue
                }

                // Store embedding locally
                let embCreatedAt = ISO8601DateFormatter().string(from: Date())
                LocalStore.shared.upsertEmbedding(imageId: imageId, keyId: keyId, embedding: embedding, createdAt: embCreatedAt, syncStatus: "pending")

                // Push to server in background
                let cropJpeg = cropToJpeg(crop)
                Task {
                    await SyncService.shared.pushEmbedding(keyId: keyId, imageId: imageId, cropJpeg: cropJpeg, embedding: embedding)
                }
            }
        } catch {
            errorMessage = "Failed to save key: \(error.localizedDescription)"
        }
    }

    private func cgImageToPixelBuffer(_ cgImage: CGImage) -> CVPixelBuffer? {
        let w = cgImage.width, h = cgImage.height
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, w, h, kCVPixelFormatType_32BGRA, nil, &pixelBuffer)
        guard let buffer = pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: base, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    private func cropToJpeg(_ crop: CGImage) -> Data {
        UIImage(cgImage: crop).jpegData(compressionQuality: 0.85) ?? Data()
    }
}
