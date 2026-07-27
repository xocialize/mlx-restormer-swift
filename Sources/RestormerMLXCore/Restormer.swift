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

    /// Loads converted safetensors weights under the strict verifier.
    public func loadWeights(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
        eval(self)
    }
}
