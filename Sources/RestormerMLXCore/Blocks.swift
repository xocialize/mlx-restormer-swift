//
//  Blocks.swift
//  mlx-restormer-swift / RestormerMLXCore
//
//  Restormer's building blocks, mirroring `basicsr/models/archs/restormer_arch.py`.
//  NHWC throughout; module keys match the upstream state dict.
//
//  Upstream: https://github.com/swz30/Restormer (MIT)
//  Paper:    Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image
//            Restoration", CVPR 2022.
//

import Foundation
import MLX
import MLXNN

// MARK: - Pixel shuffle

/// `PixelShuffle(r)` for NHWC: `(B, H, W, C·r²) → (B, H·r, W·r, C)`.
///
/// MLX has no native pixel shuffle, but it is a pure reshape + transpose. The subtlety is the
/// channel ordering: PyTorch indexes the input channel as `c·r² + i·r + j` where `(i, j)` is the
/// sub-pixel offset, so splitting the trailing axis into `(C, r, r)` lands `[c, i, j]` in exactly
/// that order. Splitting it the other way (`(r, r, C)`) compiles, runs, and silently scrambles the
/// image — the classic shape-safe failure.
public func pixelShuffle(_ x: MLXArray, _ r: Int = 2) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    precondition(c % (r * r) == 0, "pixelShuffle: \(c) channels not divisible by \(r * r)")
    return x.reshaped(b, h, w, c / (r * r), r, r)
        .transposed(0, 1, 4, 2, 5, 3)     // (B, H, r, W, r, C)
        .reshaped(b, h * r, w * r, c / (r * r))
}

/// `PixelUnshuffle(r)` for NHWC: `(B, H, W, C) → (B, H/r, W/r, C·r²)`. Inverse of ``pixelShuffle``.
public func pixelUnshuffle(_ x: MLXArray, _ r: Int = 2) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    precondition(h % r == 0 && w % r == 0, "pixelUnshuffle: \(h)x\(w) not divisible by \(r)")
    return x.reshaped(b, h / r, r, w / r, r, c)
        .transposed(0, 1, 3, 5, 2, 4)     // (B, H/r, W/r, C, r, r)
        .reshaped(b, h / r, w / r, c * r * r)
}

// MARK: - Normalization

/// Channel-dimension LayerNorm, both upstream variants.
///
/// Unlike the sibling FFTformer port — where `BiasFree` was dead code — **both variants are live
/// here**: the deblur checkpoints are `WithBias`, the Gaussian-denoising ones are `BiasFree`.
///
/// ⚠️ `BiasFree` does **not** centre the output. Upstream computes the variance (which internally
/// uses the mean) but then returns `x / sqrt(var + eps) · weight` — no mean subtraction, no bias.
/// Implementing it as "LayerNorm minus the bias term" would be wrong.
public final class ChannelLayerNorm: Module, UnaryLayer, @unchecked Sendable {
    public enum Kind: String, Sendable { case withBias, biasFree }

    @ModuleInfo(key: "body") public var body: Body
    private let kind: Kind

    public final class Body: Module, @unchecked Sendable {
        @ParameterInfo(key: "weight") public var weight: MLXArray
        @ParameterInfo(key: "bias") public var bias: MLXArray?

        public init(_ dim: Int, withBias: Bool) {
            self._weight.wrappedValue = MLXArray.ones([dim])
            self._bias.wrappedValue = withBias ? MLXArray.zeros([dim]) : nil
        }
    }

    public init(_ dim: Int, kind: Kind = .withBias) {
        self.kind = kind
        self._body.wrappedValue = Body(dim, withBias: kind == .withBias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        let inv = MLX.rsqrt(variance + 1e-5)
        switch kind {
        case .biasFree:
            return x * inv * body.weight          // uncentered, on purpose
        case .withBias:
            return centered * inv * body.weight + (body.bias ?? MLXArray(Float(0)))
        }
    }
}

// MARK: - GDFN

/// Gated-Dconv Feed-Forward Network: expand, depthwise, split, gate one half with GELU, project.
public final class FeedForward: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "project_in") public var projectIn: Conv2d
    @ModuleInfo(key: "dwconv") public var dwconv: Conv2d
    @ModuleInfo(key: "project_out") public var projectOut: Conv2d

    public init(dim: Int, ffnExpansionFactor: Float, bias: Bool) {
        let hidden = Int(Float(dim) * ffnExpansionFactor)     // int() truncation, as upstream
        self._projectIn.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: hidden * 2, kernelSize: 1, bias: bias)
        self._dwconv.wrappedValue = Conv2d(
            inputChannels: hidden * 2, outputChannels: hidden * 2, kernelSize: 3,
            padding: 1, groups: hidden * 2, bias: bias)
        self._projectOut.wrappedValue = Conv2d(
            inputChannels: hidden, outputChannels: dim, kernelSize: 1, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = dwconv(projectIn(x)).split(parts: 2, axis: -1)
        return projectOut(gelu(parts[0]) * parts[1])
    }
}

// MARK: - MDTA

/// Multi-Dconv Head Transposed Self-Attention — attention across the **channel** dimension.
///
/// Because the attention matrix is `(C/heads)²` rather than `(H·W)²`, cost is linear in pixels and
/// there is no windowing or cyclic shift to reproduce. `q`/`k` are L2-normalized along the spatial
/// axis and scaled by a learned per-head temperature (which replaces the usual `1/√d`).
///
/// ⚠️ This is also the model's known weakness: attention is spatially **global**, so it is weak on
/// *spatially varying* blur. Worth remembering when reading its defocus numbers.
public final class Attention: Module, UnaryLayer, @unchecked Sendable {
    @ParameterInfo(key: "temperature") public var temperature: MLXArray
    @ModuleInfo(key: "qkv") public var qkv: Conv2d
    @ModuleInfo(key: "qkv_dwconv") public var qkvDW: Conv2d
    @ModuleInfo(key: "project_out") public var projectOut: Conv2d

    private let heads: Int

    public init(dim: Int, heads: Int, bias: Bool) {
        self.heads = heads
        self._temperature.wrappedValue = MLXArray.ones([heads, 1, 1])
        self._qkv.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim * 3, kernelSize: 1, bias: bias)
        self._qkvDW.wrappedValue = Conv2d(
            inputChannels: dim * 3, outputChannels: dim * 3, kernelSize: 3,
            padding: 1, groups: dim * 3, bias: bias)
        self._projectOut.wrappedValue = Conv2d(
            inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: bias)
    }

    /// `(B,H,W,C)` → `(B, heads, C/heads, H·W)`.
    ///
    /// Upstream's `rearrange('b (head c) h w -> b head c (h w)')` splits the CHANNEL axis with head
    /// outermost. In NHWC that is a reshape to `(B, H·W, heads, c)` then a transpose — routing via
    /// NCHW instead would interleave the head split against the spatial flattening.
    private func headSplit(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, h * w, heads, c / heads).transposed(0, 2, 3, 1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let parts = qkvDW(qkv(x)).split(parts: 3, axis: -1)
        let q = headSplit(parts[0]), k = headSplit(parts[1]), v = headSplit(parts[2])

        func l2(_ t: MLXArray) -> MLXArray {
            t / MLX.sqrt(MLX.sum(t * t, axis: -1, keepDims: true) + 1e-12)
        }
        let attn = MLX.softmax(
            MLX.matmul(l2(q), l2(k).transposed(0, 1, 3, 2)) * temperature, axis: -1)

        let out = MLX.matmul(attn, v)                      // (B, heads, c/heads, H·W)
        return projectOut(out.transposed(0, 3, 1, 2).reshaped(b, h, w, c))
    }
}

// MARK: - Transformer block

public final class TransformerBlock: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "norm1") public var norm1: ChannelLayerNorm
    @ModuleInfo(key: "attn") public var attn: Attention
    @ModuleInfo(key: "norm2") public var norm2: ChannelLayerNorm
    @ModuleInfo(key: "ffn") public var ffn: FeedForward

    public init(dim: Int, heads: Int, ffnExpansionFactor: Float, bias: Bool,
                normKind: ChannelLayerNorm.Kind) {
        self._norm1.wrappedValue = ChannelLayerNorm(dim, kind: normKind)
        self._attn.wrappedValue = Attention(dim: dim, heads: heads, bias: bias)
        self._norm2.wrappedValue = ChannelLayerNorm(dim, kind: normKind)
        self._ffn.wrappedValue = FeedForward(
            dim: dim, ffnExpansionFactor: ffnExpansionFactor, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = x + attn(norm1(x))
        return a + ffn(norm2(a))
    }
}

// MARK: - Resampling

/// `Conv2d(n → n/2) → PixelUnshuffle(2)`, net effect `n → 2n` channels at half resolution.
public final class Downsample: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "body") public var body: [Module]     // [Conv2d, PixelUnshuffle]

    public init(_ nFeat: Int) {
        self._body.wrappedValue = [
            Conv2d(inputChannels: nFeat, outputChannels: nFeat / 2, kernelSize: 3,
                   padding: 1, bias: false)
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let conv = body[0] as? Conv2d else { return x }
        return pixelUnshuffle(conv(x), 2)
    }
}

/// `Conv2d(n → 2n) → PixelShuffle(2)`, net effect `n → n/2` channels at double resolution.
public final class Upsample: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "body") public var body: [Module]     // [Conv2d, PixelShuffle]

    public init(_ nFeat: Int) {
        self._body.wrappedValue = [
            Conv2d(inputChannels: nFeat, outputChannels: nFeat * 2, kernelSize: 3,
                   padding: 1, bias: false)
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard let conv = body[0] as? Conv2d else { return x }
        return pixelShuffle(conv(x), 2)
    }
}

/// 3×3 conv stem, zero-padded (not reflection or replication).
public final class OverlapPatchEmbed: Module, UnaryLayer, @unchecked Sendable {
    @ModuleInfo(key: "proj") public var proj: Conv2d

    public init(inChannels: Int = 3, embedDim: Int = 48, bias: Bool = false) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: embedDim,
            kernelSize: 3, padding: 1, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}
