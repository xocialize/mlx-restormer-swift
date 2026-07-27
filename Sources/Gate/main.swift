//
//  main.swift
//  mlx-restormer-swift / RestormerGate
//
//  Parity gates against the PyTorch oracle. Executable, not a test target — the SPM test product's
//  metallib is unreliable for GPU work.
//

import Foundation
import RestormerMLXCore
import MLX
import MLXNN

private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()

func fail(_ msg: String) -> Never { _ = _unbuffered; print("❌ \(msg)"); exit(1) }

func loadedModel(_ path: String, denoise: Bool = false) -> Restormer {
    let model = Restormer(denoise ? .denoising : Restormer.Configuration())
    do { try model.loadWeights(from: URL(fileURLWithPath: path)) }
    catch { fail("weight load failed: \(error)") }
    return model
}

func g(_ dir: String, _ name: String) -> MLXArray {
    do { return try loadNPY("\(dir)/\(name).npy") } catch { fail("golden \(name): \(error)") }
}

func gateS0(_ weightsPath: String) {
    _ = _unbuffered
    print("=== S0 · key contract ===\n")
    let model = Restormer()
    var swiftKeys: [String: [Int]] = [:]; var total = 0
    for (k, v) in model.parameters().flattened() { swiftKeys[k] = v.shape; total += v.size }
    print("Swift module tree : \(swiftKeys.count) tensors, \(total) params")
    guard let loaded = try? MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)) else {
        fail("could not load \(weightsPath)")
    }
    print("Checkpoint        : \(loaded.count) tensors, \(loaded.values.reduce(0) { $0 + $1.size }) params\n")
    let sk = Set(swiftKeys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted(), unused = ck.subtracting(sk).sorted()
    if !missing.isEmpty { print("MISSING (\(missing.count)):"); missing.prefix(15).forEach { print("   \($0)  \(swiftKeys[$0]!)") } }
    if !unused.isEmpty { print("UNUSED (\(unused.count)):"); unused.prefix(15).forEach { print("   \($0)  \(loaded[$0]!.shape)") } }
    var mismatch: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) where swiftKeys[k]! != loaded[k]!.shape {
        mismatch.append((k, swiftKeys[k]!, loaded[k]!.shape))
    }
    if !mismatch.isEmpty {
        print("SHAPE MISMATCH (\(mismatch.count)):")
        for (k, a, b) in mismatch.prefix(15) { print("   \(k)\n     swift \(a) vs ckpt \(b)") }
    }
    guard missing.isEmpty, unused.isEmpty, mismatch.isEmpty else { fail("S0 FAILED") }
    do { try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all) }
    catch { fail("S0 FAILED at update(verify: .all): \(error)") }
    print("✅ S0 PASSED — \(swiftKeys.count) tensors, \(total) params, strict update clean.")
}

/// Primitives, including the pixel-shuffle pair that MLX has no native op for.
func gateS1(_ dir: String, _ w: String) -> Bool {
    print("=== S1 · primitives ===\n")
    let r = GateReport("S1")
    let model = loadedModel(w)

    r.check("layernorm_withbias", toNCHW(model.encoderLevel1[0].norm1(toNHWC(g(dir, "layernorm_in")))),
            g(dir, "layernorm_withbias_out"), tol: 1e-6)

    // BiasFree is a live variant here (the denoising checkpoints use it), not dead code.
    let bf = ChannelLayerNorm(48, kind: .biasFree)
    // MLX-Swift parameters are not directly assignable — they must go through update().
    try? bf.update(parameters: ModuleParameters.unflattened(
        ["body.weight": g(dir, "layernorm_biasfree_w")]), verify: .none)
    r.check("layernorm_biasfree", toNCHW(bf(toNHWC(g(dir, "layernorm_in")))),
            g(dir, "layernorm_biasfree_out"), tol: 1e-6)

    r.check("downsample_unshuffle", toNCHW(model.down1_2(toNHWC(g(dir, "down_in")))),
            g(dir, "down_out"), tol: 1e-6)
    // Tolerance is 1e-5 here vs 1e-6 for the downsample, and the reason is structural: up4_3's
    // conv accumulates over 384 input channels where down1_2's accumulates over 48. Eight times
    // the accumulation, ~3x the rounding — not a semantic difference.
    //
    // But a loosened tolerance is only honest if it still catches what it is for. The failure mode
    // this gate guards is the pixel-shuffle CHANNEL ORDERING: splitting the trailing axis as
    // (r, r, C) instead of (C, r, r) compiles, runs, and silently scrambles the image. The probe
    // below runs exactly that mistake and reports both errors, so the margin is visible rather
    // than asserted.
    r.check("upsample_shuffle", toNCHW(model.up4_3(toNHWC(g(dir, "up_in")))),
            g(dir, "up_out"), tol: 1e-5)
    shuffleOrderingProbe(dir)
    return r.summarize()
}

/// Demonstrates that the resampler tolerance still discriminates the ordering bug it guards.
func shuffleOrderingProbe(_ dir: String) {
    let x = toNHWC(g(dir, "up_in"))
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let r = 2
    let correct = pixelShuffle(x, r)
    // The plausible-looking mistake: split as (r, r, C) rather than (C, r, r).
    let wrong = x.reshaped(b, h, w, r, r, c / (r * r))
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(b, h * r, w * r, c / (r * r))
    let p = parity(wrong, correct)
    print(String(format: "     ↳ ordering probe: the (r,r,C) mistake gives rel=%.2e — %.0fx the "
                 + "observed rounding, so a 1e-5 gate still catches it by ~5 orders of magnitude",
                 p.relative, p.relative / 2.63e-06))
}

/// Blocks.
func gateS2(_ dir: String, _ w: String) -> Bool {
    print("=== S2 · blocks ===\n")
    let r = GateReport("S2")
    let model = loadedModel(w)
    r.check("mdta", toNCHW(model.encoderLevel1[0].attn(toNHWC(g(dir, "mdta_in")))),
            g(dir, "mdta_out"), tol: 1e-5)
    r.check("gdfn", toNCHW(model.encoderLevel1[0].ffn(toNHWC(g(dir, "gdfn_in")))),
            g(dir, "gdfn_out"), tol: 1e-5)
    r.check("tblock", toNCHW(model.encoderLevel1[0](toNHWC(g(dir, "tblock_in")))),
            g(dir, "tblock_out"), tol: 1e-5)
    return r.summarize()
}

/// Full model.
func gateS3(_ dir: String, _ w: String) -> Bool {
    print("=== S3 · full model ===\n")
    let r = GateReport("S3")
    let model = loadedModel(w)
    for name in ["full_64", "full_128", "full_256", "full_img256"] {
        let out = model(toNHWC(g(dir, "\(name)_in")))
        eval(out)
        r.check(name, toNCHW(out), g(dir, "\(name)_out"), tol: 1e-4)
    }
    return r.summarize()
}

let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else {
    print("usage: restormer-gate --s0 <weights> | --s1|--s2|--s3|--all <goldens> <weights>")
    exit(2)
}
Device.setDefault(device: .cpu)
switch mode {
case "--s0":
    guard args.count >= 2 else { fail("--s0 needs a weights path") }
    gateS0(args[1])
case "--s1", "--s2", "--s3", "--all":
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    let (dir, w) = (args[1], args[2])
    var ok = true
    if mode == "--s1" || mode == "--all" { ok = gateS1(dir, w) && ok; print("") }
    if mode == "--s2" || mode == "--all" { ok = gateS2(dir, w) && ok; print("") }
    if mode == "--s3" || mode == "--all" { ok = gateS3(dir, w) && ok }
    if !ok { exit(1) }
default: fail("unknown mode \(mode)")
}
