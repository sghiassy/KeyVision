import SwiftUI

struct RecognitionOverlay: View {
    let result: MatchResult?

    var body: some View {
        VStack {
            Spacer()
            if let result, result.confidence != .noMatch {
                HStack(spacing: 12) {
                    Circle()
                        .fill(result.confidence == .high ? Color.green : Color.yellow)
                        .frame(width: 10, height: 10)
                    Text(result.label)
                        .font(.headline)
                        .foregroundColor(.white)
                    if result.confidence == .high {
                        NavigationLink("View Details") {
                            KeyDetailView(keyId: result.keyId, label: result.label)
                        }
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.85))
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(.ultraThinMaterial)
                .clipShape(Capsule())
                .padding(.bottom, 48)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .animation(.easeInOut(duration: 0.25), value: result?.keyId)
    }
}
