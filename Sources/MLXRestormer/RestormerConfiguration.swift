import Foundation
import MLXToolKit
import RestormerMLXCore

/// A Restormer checkpoint. **One architecture, three products** — this is the row's whole thesis.
///
/// Each is a genuinely different restoration job, not a quality tier, so the choice is the caller's
/// (or the planner's) and never a fallback.
public enum RestormerVariant: String, Codable, Sendable, CaseIterable {
    /// **Motion deblur** — GoPro 32.92. The general camera-shake / subject-motion case.
    case motionDeblur
    /// **Single-image defocus deblur** — DPDD 25.98. Worth having specifically because the entire
    /// high end of this task is licence-poisoned: IFAN, KPAC, DRBNet and LaKDNet are all AGPL-3.0.
    /// Restormer is 0.44 dB off the best and is the only shippable option.
    ///
    /// ⚠️ Not the `dual_pixel_defocus` checkpoint, deliberately. That one needs dual-pixel sensor
    /// data user-imported photos never carry, and its ~0.7 dB-higher numbers are the ones most
    /// often quoted.
    case defocusDeblur
    /// **Real denoising** — SIDD/DND. Carries the only solid **DND 40.03**, i.e. it generalizes
    /// across sensors rather than to SIDD's five smartphone cameras.
    case realDenoise

    public var repo: String {
        switch self {
        case .motionDeblur: return "mlx-community/Restormer-motion-deblurring-fp32"
        case .defocusDeblur: return "mlx-community/Restormer-defocus-deblurring-fp32"
        case .realDenoise: return "mlx-community/Restormer-real-denoising-fp32"
        }
    }

    /// 🔑 **This is load-bearing, not cosmetic.** The deblur checkpoints normalize `WithBias`; the
    /// denoising ones are `BiasFree` and therefore have **no bias vectors at all** — 406 tensors /
    /// 26,111,668 params against the deblur models' 494 / 26,126,644. Choosing wrong fails the
    /// strict load with exactly 88 missing keys, which is the good outcome.
    public var normKind: ChannelLayerNorm.Kind {
        switch self {
        case .motionDeblur, .defocusDeblur: return .withBias
        case .realDenoise: return .biasFree
        }
    }

    /// 104 MB at fp32 is small enough that a lower dtype buys little, and restoration is
    /// precision-sensitive. Measure before changing this.
    public var quant: Quant { .fp32 }

    var coreConfiguration: Restormer.Configuration {
        var c = Restormer.Configuration()
        c.normKind = normKind
        return c
    }
}

/// Init-time configuration for `RestormerRestorePackage` (C9).
public struct RestormerConfiguration: PackageConfiguration, ModelStorable {
    public var variant: RestormerVariant

    /// Tile extent for the internal tiled path. Rounded down to a multiple of 8 by the core.
    ///
    /// Tiling is **mandatory**, not an optimization: full-frame costs 15.50 GB MLX / 48.02 GB phys
    /// at 1080p (measured). Tiled, the MLX peak is flat at ~2.6 GB from 512² to 1080p, because the
    /// peak is one-tile-sized and a bigger image simply runs more tiles.
    public var tile: Int

    /// Context pixels per tile side, discarded into the feathered blend.
    ///
    /// Measured on seam visibility rather than PSNR: **overlap 0 leaves a faint but real seam**
    /// (boundary gradient 1.31× the interior), while every aligned overlap ≥ 8 measures clean
    /// (1.09–1.20×). Note PSNR-against-full-frame *prefers* overlap 0 — which is exactly why it is
    /// the wrong metric for a tiler.
    public var overlap: Int

    public var modelsRootDirectory: URL?
    public var weightsURL: URL?

    public init(variant: RestormerVariant = .motionDeblur,
                tile: Int = 384,
                overlap: Int = 32,
                modelsRootDirectory: URL? = nil,
                weightsURL: URL? = nil) {
        self.variant = variant
        self.tile = tile
        self.overlap = overlap
        self.modelsRootDirectory = modelsRootDirectory
        self.weightsURL = weightsURL
    }

    private enum CodingKeys: String, CodingKey {
        case variant, tile, overlap
    }
}

extension RestormerConfiguration: QuantConfigured {
    public var quant: Quant { variant.quant }
}

extension RestormerConfiguration: WeightSourcing {
    public var weightSources: [WeightSource] {
        [WeightSource(role: "weights", repo: variant.repo, revision: nil,
                      matching: ["model.safetensors"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let weightsURL, FileManager.default.fileExists(atPath: weightsURL.path) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }
}
