//
//  Restormer.swift
//  mlx-restormer-swift / RestormerMLXCore
//
//  Role: MLX-Swift port of Restormer — one architecture, several restoration products.
//        Motion deblur (GoPro 32.92) · single-image defocus deblur (DPDD 25.98, the only
//        permissive competitive option) · real denoising (DND 40.03, cross-sensor).
//
//  Upstream: https://github.com/swz30/Restormer — plain **MIT** (Copyright 2022 Syed Waqas Zamir).
//            Not to be confused with the same author's MIRNet / MPRNet / CycleISP, which carry an
//            Academic Public License.
//  Paper:    Zamir et al., CVPR 2022. 26,126,644 parameters.
//
//  Conventions: NHWC; module keys mirror the upstream state dict exactly.
//

import Foundation
import MLX
import MLXNN

public final class Restormer: Module, @unchecked Sendable {

    public struct Configuration: Sendable {
        public var inpChannels = 3
        public var outChannels = 3
        public var dim = 48
        public var numBlocks = [4, 6, 6, 8]
        public var numRefinementBlocks = 4
        public var heads = [1, 2, 4, 8]
        public var ffnExpansionFactor: Float = 2.66
        public var bias = false
        /// **Live choice, not dead code.** The deblur checkpoints are `WithBias`; the Gaussian
        /// denoising checkpoints are `BiasFree`. Picking the wrong one changes the key set (the
        /// bias vectors appear or vanish) and fails the strict load — which is the good outcome.
        public var normKind: ChannelLayerNorm.Kind = .withBias

        /// The released deblur + real-denoise configs are all defaults.
        public init() {}

        /// The Gaussian-denoising configuration.
        public static var denoising: Configuration {
            var c = Configuration()
            c.normKind = .biasFree
            return c
        }
    }

    @ModuleInfo(key: "patch_embed") public var patchEmbed: OverlapPatchEmbed
    @ModuleInfo(key: "encoder_level1") public var encoderLevel1: [TransformerBlock]
    @ModuleInfo(key: "down1_2") public var down1_2: Downsample
    @ModuleInfo(key: "encoder_level2") public var encoderLevel2: [TransformerBlock]
    @ModuleInfo(key: "down2_3") public var down2_3: Downsample
    @ModuleInfo(key: "encoder_level3") public var encoderLevel3: [TransformerBlock]
    @ModuleInfo(key: "down3_4") public var down3_4: Downsample
    @ModuleInfo(key: "latent") public var latent: [TransformerBlock]
    @ModuleInfo(key: "up4_3") public var up4_3: Upsample
    @ModuleInfo(key: "reduce_chan_level3") public var reduceChan3: Conv2d
    @ModuleInfo(key: "decoder_level3") public var decoderLevel3: [TransformerBlock]
    @ModuleInfo(key: "up3_2") public var up3_2: Upsample
    @ModuleInfo(key: "reduce_chan_level2") public var reduceChan2: Conv2d
    @ModuleInfo(key: "decoder_level2") public var decoderLevel2: [TransformerBlock]
    @ModuleInfo(key: "up2_1") public var up2_1: Upsample
    /// Level 1's decoder runs at **2·dim**, not `dim` — there is deliberately no 1×1 reduction
    /// after `up2_1`, so the concatenated skip stays at full width. Sizing this at `dim` is the
    /// obvious-looking mistake.
    @ModuleInfo(key: "decoder_level1") public var decoderLevel1: [TransformerBlock]
    @ModuleInfo(key: "refinement") public var refinement: [TransformerBlock]
    @ModuleInfo(key: "output") public var output: Conv2d

    public init(_ cfg: Configuration = Configuration()) {
        let d = cfg.dim
        let f = cfg.ffnExpansionFactor
        let b = cfg.bias
        let n = cfg.normKind

        func blocks(_ count: Int, _ dim: Int, _ heads: Int) -> [TransformerBlock] {
            (0 ..< count).map { _ in
                TransformerBlock(dim: dim, heads: heads, ffnExpansionFactor: f, bias: b, normKind: n)
            }
        }

        self._patchEmbed.wrappedValue = OverlapPatchEmbed(
            inChannels: cfg.inpChannels, embedDim: d, bias: b)
        self._encoderLevel1.wrappedValue = blocks(cfg.numBlocks[0], d, cfg.heads[0])
        self._down1_2.wrappedValue = Downsample(d)
        self._encoderLevel2.wrappedValue = blocks(cfg.numBlocks[1], d * 2, cfg.heads[1])
        self._down2_3.wrappedValue = Downsample(d * 2)
        self._encoderLevel3.wrappedValue = blocks(cfg.numBlocks[2], d * 4, cfg.heads[2])
        self._down3_4.wrappedValue = Downsample(d * 4)
        self._latent.wrappedValue = blocks(cfg.numBlocks[3], d * 8, cfg.heads[3])

        self._up4_3.wrappedValue = Upsample(d * 8)
        self._reduceChan3.wrappedValue = Conv2d(
            inputChannels: d * 8, outputChannels: d * 4, kernelSize: 1, bias: b)
        self._decoderLevel3.wrappedValue = blocks(cfg.numBlocks[2], d * 4, cfg.heads[2])

        self._up3_2.wrappedValue = Upsample(d * 4)
        self._reduceChan2.wrappedValue = Conv2d(
            inputChannels: d * 4, outputChannels: d * 2, kernelSize: 1, bias: b)
        self._decoderLevel2.wrappedValue = blocks(cfg.numBlocks[1], d * 2, cfg.heads[1])

        self._up2_1.wrappedValue = Upsample(d * 2)
        self._decoderLevel1.wrappedValue = blocks(cfg.numBlocks[0], d * 2, cfg.heads[0])
        self._refinement.wrappedValue = blocks(cfg.numRefinementBlocks, d * 2, cfg.heads[0])

        self._output.wrappedValue = Conv2d(
            inputChannels: d * 2, outputChannels: cfg.outChannels,
            kernelSize: 3, padding: 1, bias: b)
    }

    /// Forward on an NHWC tensor whose H and W are multiples of 8 (three ÷2 stages).
    /// Use ``restore(_:)`` for the padding contract.
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let e1 = encoderLevel1.reduce(patchEmbed(input)) { $1($0) }
        let e2 = encoderLevel2.reduce(down1_2(e1)) { $1($0) }
        let e3 = encoderLevel3.reduce(down2_3(e2)) { $1($0) }
        let lat = latent.reduce(down3_4(e3)) { $1($0) }

        var d3 = concatenated([up4_3(lat), e3], axis: -1)
        d3 = reduceChan3(d3)
        d3 = decoderLevel3.reduce(d3) { $1($0) }

        var d2 = concatenated([up3_2(d3), e2], axis: -1)
        d2 = reduceChan2(d2)
        d2 = decoderLevel2.reduce(d2) { $1($0) }

        // No channel reduction here — the concat stays at 2·dim through level 1 and refinement.
        var d1 = concatenated([up2_1(d2), e1], axis: -1)
        d1 = decoderLevel1.reduce(d1) { $1($0) }
        d1 = refinement.reduce(d1) { $1($0) }

        return output(d1) + input      // global residual
    }

    /// Reflect-pads to a multiple of 8, runs, crops back. Three ÷2 stages mean the input must be
    /// divisible by 8; upstream's own test scripts pad the same way.
    public func restore(_ image: MLXArray) -> MLXArray {
        let (h, w) = (image.dim(1), image.dim(2))
        let ph = (8 - h % 8) % 8
        let pw = (8 - w % 8) % 8
        guard ph != 0 || pw != 0 else { return self(image) }

        var padded = image
        if ph > 0 {
            let idx = MLXArray((1...ph).map { Int32(h - 1 - $0) })
            padded = concatenated([padded, padded.take(idx, axis: 1)], axis: 1)
        }
        if pw > 0 {
            let idx = MLXArray((1...pw).map { Int32(w - 1 - $0) })
            padded = concatenated([padded, padded.take(idx, axis: 2)], axis: 2)
        }
        return self(padded)[0..., 0 ..< h, 0 ..< w, 0...]
    }

    /// Tiled restoration — **the production path.** Full-frame is not viable above ~512².
    ///
    /// Measured (M5 Max, `--bench`): 512² costs 3.26 GB MLX / 12.96 GB phys; 1080p costs
    /// **15.50 GB MLX / 48.02 GB phys**. A 104 MB model must not need 48 GB to deblur one frame.
    /// (The sibling FFTformer port is worse still at ~40 GB MLX, for the same reason: level 1 runs
    /// at full resolution with a wide channel expansion.)
    ///
    /// 🔑 **Tile geometry is aligned to 8.** Three `pixelUnshuffle(2)` stages mean the 2×2 grouping
    /// grid is measured from the tile origin at strides 2, 4 and 8 in full-resolution pixels, so a
    /// tile origin that is not ≡ 0 (mod 8) shifts that grid out of phase with the full-frame
    /// decomposition. Same class of bug as FFTformer's 32-alignment, one third the stride. Both
    /// tile and overlap are rounded down to a multiple of 8.
    ///
    /// Feathered blend rather than crop-valid: the receptive field of a 4-level UNet with 4+6+6+8
    /// transformer blocks far exceeds any practical overlap, so blending removes seams without
    /// pretending to reproduce a full-frame result that is itself unattainable at these sizes.
    ///
    /// - Parameter onTile: called as `(completed, total)` before each tile — the seam for
    ///   cooperative cancellation and progress. Throwing aborts, and the error propagates unchanged.
    public func restoreTiled(_ image: MLXArray, tile: Int = 384, overlap: Int = 32,
                             onTile: ((Int, Int) throws -> Void)? = nil) rethrows -> MLXArray {
        let tile = max(64, (tile / 8) * 8)
        let overlap = (overlap / 8) * 8
        precondition(tile > 2 * overlap, "tile (\(tile)) must exceed 2·overlap (\(2 * overlap))")

        let (b, h, w, c) = (image.dim(0), image.dim(1), image.dim(2), image.dim(3))
        if h <= tile && w <= tile { return restore(image) }

        let step = tile - 2 * overlap
        var acc = MLXArray.zeros([b, h, w, c], dtype: .float32)
        var wsum = MLXArray.zeros([1, h, w, 1], dtype: .float32)

        let total = ((h + step - 1) / step) * ((w + step - 1) / step)
        var done = 0

        for coreY in stride(from: 0, to: h, by: step) {
            let inY0 = max(0, coreY - overlap)
            let inY1 = min(h, coreY + step + overlap)
            for coreX in stride(from: 0, to: w, by: step) {
                try onTile?(done, total)
                done += 1

                let inX0 = max(0, coreX - overlap)
                let inX1 = min(w, coreX + step + overlap)

                let restored = restore(image[0..., inY0 ..< inY1, inX0 ..< inX1, 0...])
                let weight = Self.featherWeights(height: inY1 - inY0, width: inX1 - inX0,
                                                 ramp: overlap,
                                                 topEdge: inY0 == 0, bottomEdge: inY1 == h,
                                                 leftEdge: inX0 == 0, rightEdge: inX1 == w)
                acc[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] =
                    acc[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] + restored.asType(.float32) * weight
                wsum[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] =
                    wsum[0..., inY0 ..< inY1, inX0 ..< inX1, 0...] + weight

                // Realize and release per tile — MLX otherwise accumulates unbounded residency
                // across a long sequential graph, which is exactly what a tile loop is.
                eval(acc, wsum)
                MLX.Memory.clearCache()
            }
        }
        return clip(acc / maximum(wsum, MLXArray(1e-8)), min: 0, max: 1).asType(image.dtype)
    }

    /// Separable linear ramp; image-boundary edges are left unramped, since nothing overlaps them
    /// and a falloff there would divide by a small weight and amplify noise at the frame border.
    private static func featherWeights(height: Int, width: Int, ramp: Int,
                                       topEdge: Bool, bottomEdge: Bool,
                                       leftEdge: Bool, rightEdge: Bool) -> MLXArray {
        func profile(_ n: Int, _ startFlat: Bool, _ endFlat: Bool) -> [Float] {
            var v = [Float](repeating: 1, count: n)
            guard ramp > 0 else { return v }
            let r = min(ramp, n / 2)
            for i in 0 ..< r {
                let t = (Float(i) + 0.5) / Float(r)
                if !startFlat { v[i] = t }
                if !endFlat { v[n - 1 - i] = min(v[n - 1 - i], t) }
            }
            return v
        }
        let y = MLXArray(profile(height, topEdge, bottomEdge), [1, height, 1, 1])
        let x = MLXArray(profile(width, leftEdge, rightEdge), [1, 1, width, 1])
        return y * x
    }

    /// Loads converted safetensors weights under the strict verifier.
    public func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}
