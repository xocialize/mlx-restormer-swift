import Foundation
import CoreGraphics
import CoreImage
import CoreVideo
import ImageIO
import UniformTypeIdentifiers
import MLX
import MLXToolKit
import MLXProfiling
import Hub
import RestormerMLXCore

/// Errors at the Restormer package boundary.
public enum RestormerPackageError: Error, Equatable {
    case imageDecodeFailed(String)
    case imageEncodeFailed
    case weightsMissing(String)
}

/// An MLXEngine `imageRestore` package over **Restormer** — one architecture, three restoration
/// products, selected by ``RestormerVariant``.
///
/// The **third** package on `imageRestore`, alongside NAFNet and FFTformer. Deliberately not a new
/// capability: same request, same response, same canonical output — a different backer chosen by
/// `PackageID`. This is the "capability redundancy is a feature" pattern from `MODEL-CATALOG.md`.
///
/// What each variant is *for*, since they are different jobs rather than quality tiers:
/// - `.motionDeblur` — camera shake / subject motion (GoPro 32.92)
/// - `.defocusDeblur` — out-of-focus (DPDD 25.98). Notable because the high end of this task is
///   licence-poisoned (IFAN / KPAC / DRBNet / LaKDNet are all AGPL-3.0); this is the only
///   shippable option.
/// - `.realDenoise` — the only solid **DND 40.03**, i.e. cross-sensor denoise generalization
///
/// ⚠️ Benchmark rank is a weak predictor of real-world value. Validation needs corpora **C4**
/// (focus brackets) and **C5** (ISO brackets) — see `mlxengine-todo/CORPUS-NEEDS.md`. Restormer's
/// attention is spatially **global**, so it is weak on *spatially varying* blur; a flat test chart
/// would hide exactly that.
@InferenceActor
public final class RestormerRestorePackage: ModelPackage {
    public typealias Configuration = RestormerConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // kkkls/Restormer is MIT (LICENSE + every source header); the weights are committed
            // in-repo under that same license — the favourable case, no separate weights statement
            // to interpret. Port code MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "swz30/Restormer", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // Split footprint (engine 1.14) — ✅ MEASURED through the REAL `MLXServeEngine` via
                // `MLXEngineTestKit.ValidationHarness` (`swift run restormer-validate`), process
                // `phys_footprint`, floor read post-load/pre-run:
                //
                //   [restormer-motionDeblur] SPLIT floor=0.14GB peak=4.90GB act=4.76GB retain=0.36GB
                //                            engine=0.15GB reserve=4.50GB load=0.1s run=20.7s @1920x1080
                //
                // Declared with margin: resident 180 MB (floor 136.5 MB), activation 5.5 GB (measured
                // 4.76 GB). The earlier 4.5 GB estimate from the gate's `--bench` was slightly UNDER
                // the truth — the same direction that makes under-declaration dangerous, just by less
                // than CIDNet's 2x miss.
                //
                // Tiling is internal, so the peak is one-tile-sized and flat in resolution. Untiled
                // for comparison: 15.50 GB MLX / 48.02 GB phys at 1080p.
                footprints: [
                    QuantFootprint(quant: .fp32,
                                   residentBytes: 180_000_000,
                                   peakActivationBytes: 5_500_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageRestoreContract.descriptor(
                    name: "restormer-restore",
                    summary: "Restormer image restoration: motion deblur, single-image defocus "
                        + "deblur, or real denoise — one architecture, selected by variant."
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: Restormer?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }

        let url: URL
        if let explicit = configuration.weightsURL {
            guard FileManager.default.fileExists(atPath: explicit.path) else {
                throw RestormerPackageError.weightsMissing(explicit.path)
            }
            url = explicit
        } else {
            // Since contract 1.24 the engine materializes declared `weightSources` before load().
            // This snapshot is the defensive path — it finds everything already present in the
            // normal flow, and still works for a standalone (engine-less) consumer of the package.
            let repo = configuration.variant.repo
            let hub = configuration.modelsRootDirectory.map { HubApi(downloadBase: $0) } ?? HubApi()
            let dir = try await hub.snapshot(from: Hub.Repo(id: repo),
                                             matching: ["model.safetensors"]) { progress, speed in
                WeightDownloadProgress.report(fraction: progress.fractionCompleted, bytesPerSecond: speed)
            }
            url = dir.appendingPathComponent("model.safetensors")
        }

        // The variant picks the LayerNorm kind, which determines the KEY SET — the denoising
        // checkpoints are BiasFree and carry no bias vectors at all (406 tensors vs 494). A
        // mismatch fails the strict load with 88 missing keys rather than loading something wrong.
        let net = Restormer(configuration.variant.coreConfiguration)
        try net.loadWeights(from: url)
        model = net
    }

    public func unload() async {
        model = nil
        MLX.Memory.clearCache()   // drop the retained MLX pool so eviction frees RSS, not just refs
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint is the FIRST act of run(), before notLoaded validation.
        // Mid-run cadence: like NAFNet this is ONE monolithic full-frame eval with no iterative
        // loop, so the honest seams are pre-forward (post-decode) and post-forward (pre-encode).
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .imageRestore,
              let req = request as? ImageRestoreRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let pb = try Self.decodeToPixelBuffer(req.image)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let x = rgbNHWC(from: ensureBGRA(pb), width: w, height: h) else {
            throw RestormerPackageError.imageDecodeFailed("NHWC conversion (\(w)x\(h))")
        }

        // Pre-forward checkpoint: last seam before committing to the monolithic eval.
        try Task.checkCancellation()
        let prof = MLXProfiler.shared
        prof.beginRun("restormer imageRestore \(configuration.variant.rawValue) \(w)x\(h)")
        // TILED is the production path, not an optimization: full-frame costs 15.50 GB MLX /
        // 48.02 GB phys at 1080p (measured). Tiled, the MLX peak is flat at ~2.6 GB regardless of
        // input size, because the peak is one-tile-sized. Tile geometry is 8-aligned — three
        // pixelUnshuffle(2) stages make the grouping grid phase-sensitive to the tile origin.
        // The tile loop is a REAL iterative seam — unlike NAFNet's single monolithic forward — so
        // cancellation and progress are checkpointed per tile, not merely at the phase boundaries.
        // The CancellationError propagates unchanged so the engine can tell user-cancel from a
        // package error.
        let restoredNHWC = try prof.region("deblur", "forward") {
            try model.restoreTiled(x, tile: configuration.tile, overlap: configuration.overlap) { done, total in
                try Task.checkCancellation()
                RunProgress.report(RunPhaseReport(phase: .postprocess, step: done + 1,
                                                  totalSteps: total))
            }
        }
        let outPB = pixelBuffer(fromRGBNHWC: restoredNHWC, width: w, height: h)
        prof.endRun(denominators: ["image": 1])
        guard let outPB else { throw RestormerPackageError.imageEncodeFailed }

        // Post-forward checkpoint: between materialization and output encode.
        try Task.checkCancellation()
        let outImage: Image
        if req.image.format == .rawBGRA8 {
            guard let raw = Self.encodeRawBGRA8(outPB) else { throw RestormerPackageError.imageEncodeFailed }
            outImage = raw
        } else {
            guard let png = Self.encodePNG(outPB) else { throw RestormerPackageError.imageEncodeFailed }
            outImage = Image(format: .png, data: png, width: w, height: h)
        }
        return ImageRestoreResponse(image: outImage)
    }

    // MARK: - Image codec
    //
    // Same shape as the sibling image packages. Duplicated rather than shared: each `-swift`
    // package stays independently buildable, and the codec is the package's own boundary.

    /// Decode a canonical `Image` (.png/.jpeg/.rawBGRA8) to a BGRA `CVPixelBuffer`.
    nonisolated static func decodeToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        if image.format == .rawBGRA8 { return try rawBGRA8ToPixelBuffer(image) }
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw RestormerPackageError.imageDecodeFailed("unreadable \(image.format.rawValue) data")
        }
        let w = cg.width, h = cg.height
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RestormerPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let ctx = CGContext(
                data: base, width: w, height: h, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue) else {
            throw RestormerPackageError.imageDecodeFailed("CGContext for BGRA draw")
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buffer
    }

    /// Encode a BGRA `CVPixelBuffer` as PNG bytes.
    nonisolated static func encodePNG(_ pb: CVPixelBuffer) -> Data? {
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: [.cacheIntermediates: false])
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? out as Data : nil
    }

    /// Wrap raw interleaved BGRA8 bytes straight into a 32BGRA `CVPixelBuffer` — no decode.
    nonisolated static func rawBGRA8ToPixelBuffer(_ image: Image) throws -> CVPixelBuffer {
        guard let w = image.width, let h = image.height, w > 0, h > 0 else {
            throw RestormerPackageError.imageDecodeFailed("rawBGRA8 requires width/height")
        }
        let srcStride = image.bytesPerRow ?? (w * 4)
        guard srcStride >= w * 4, image.data.count >= srcStride * h else {
            throw RestormerPackageError.imageDecodeFailed(
                "rawBGRA8 data too small (\(image.data.count) < \(srcStride * h))")
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw RestormerPackageError.imageDecodeFailed("pixel buffer allocation (\(w)x\(h))")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw RestormerPackageError.imageDecodeFailed("pixel buffer base address")
        }
        let dstStride = CVPixelBufferGetBytesPerRow(buffer)
        let rowBytes = min(srcStride, dstStride)
        image.data.withUnsafeBytes { (src: UnsafeRawBufferPointer) in
            guard let srcBase = src.baseAddress else { return }
            for row in 0..<h {
                memcpy(base.advanced(by: row * dstStride), srcBase.advanced(by: row * srcStride), rowBytes)
            }
        }
        return buffer
    }

    /// Emit a 32BGRA `CVPixelBuffer` as tightly-packed raw BGRA8 `Image` bytes.
    nonisolated static func encodeRawBGRA8(_ pb: CVPixelBuffer) -> Image? {
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard w > 0, h > 0 else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let srcStride = CVPixelBufferGetBytesPerRow(pb)
        let dstStride = w * 4
        var out = Data(count: dstStride * h)
        out.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
            guard let dstBase = dst.baseAddress else { return }
            for row in 0..<h {
                memcpy(dstBase.advanced(by: row * dstStride), base.advanced(by: row * srcStride), dstStride)
            }
        }
        return Image.rawBGRA8(data: out, width: w, height: h)
    }
}

extension RestormerRestorePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(RestormerRestorePackage.self)
    }
}
