// ConformanceTests.swift — Restormer through the engine's offline gates (no MLX kernels run).
//
// The variant-specific tests matter more here than in the sibling packages, because this is the
// first multi-variant port where a variant changes the *key set* rather than just the weights:
// the denoising checkpoint is BiasFree and has no bias vectors at all.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
import RestormerMLXCore
@testable import MLXRestormer

final class ConformanceTests: XCTestCase {

    // MARK: - MAT

    func testMATGate() {
        let report = MaterializationConformance.check(freshConfiguration: RestormerConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesDeclaredForEveryVariant() {
        var repos = Set<String>()
        for variant in RestormerVariant.allCases {
            let sources = RestormerConfiguration(variant: variant).weightSources
            XCTAssertEqual(sources.count, 1, "\(variant)")
            XCTAssertEqual(sources[0].repo, variant.repo)
            XCTAssertEqual(sources[0].matching, ["model.safetensors"], "\(variant)")
            repos.insert(sources[0].repo)
        }
        // Each variant is a different product and must resolve to its own repo — a shared repo
        // would silently serve one checkpoint for all three.
        XCTAssertEqual(repos.count, RestormerVariant.allCases.count)
    }

    func testExplicitWeightsURLSuppressesMaterialization() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("restormer-\(UUID().uuidString).safetensors")
        try Data([0x00]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertTrue(RestormerConfiguration(weightsURL: tmp).missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(
            RestormerConfiguration(weightsURL: tmp.appendingPathExtension("nope"))
                .missingWeightSources(storeRoot: nil).count, 1)
    }

    // MARK: - CAN

    func testCANGatePreCancelledRun() async {
        let package = RestormerRestorePackage(configuration: RestormerConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageRestoreRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        let manifest = RestormerRestorePackage.manifest
        XCTAssertTrue(CancellationConformance.longRunImplied(by: manifest),
                      "4.5 GB declared activation should imply long runs")
        // run() has a REAL iterative seam — the tile loop — and checkpoints once per tile via the
        // core's onTile hook, reporting RunProgress on the same unit.
        let report = CancellationConformance.checkCadence(
            manifest: manifest,
            posture: .cadence([
                .init(phase: .postprocess, unit: .chunk, reportsRunProgress: true),
                .init(phase: .encode, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Manifest

    func testManifestSurfacesAndLicence() {
        let m = RestormerRestorePackage.manifest
        XCTAssertEqual(m.capabilities, [.imageRestore], "no new capability — same request shape")
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].name, "restormer-restore")
        XCTAssertEqual(m.license.weightLicense, .mit)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.provenance.sourceRepo, "swz30/Restormer")
    }

    func testFootprintIsSplitAndFlatInResolution() {
        guard let fp = RestormerRestorePackage.manifest.requirements.footprints
            .first(where: { $0.quant == .fp32 }) else { return XCTFail("no fp32 footprint") }
        // 26.1 M params @ fp32 = 104.5 MB; the floor must cover it without absorbing the activation.
        XCTAssertGreaterThan(fp.residentBytes, 104_500_000)
        XCTAssertLessThan(fp.residentBytes, 500_000_000)
        XCTAssertGreaterThan(fp.peakActivationBytes, fp.residentBytes)
        // Tiling makes the peak one-tile-sized, so the declaration must be far below the ~48 GB
        // an untiled 1080p frame would need. A declaration up there would mean tiling regressed.
        XCTAssertLessThan(fp.peakActivationBytes, 10_000_000_000)
    }

    func testQuantConfiguredMatchesADeclaredFootprint() {
        let declared = Set(RestormerRestorePackage.manifest.requirements.footprints.map(\.quant))
        for v in RestormerVariant.allCases {
            XCTAssertTrue(declared.contains(RestormerConfiguration(variant: v).quant), "\(v)")
        }
    }

    // MARK: - Variant semantics

    /// 🔑 The denoising variant is BiasFree and therefore has a *different key set*. Pin the mapping
    /// so a future edit cannot quietly point a denoise config at a WithBias tree — that would fail
    /// the strict load, which is the good outcome, but only if this mapping stays correct.
    func testDenoiseVariantIsBiasFreeAndDeblurIsWithBias() {
        XCTAssertEqual(RestormerVariant.realDenoise.normKind, .biasFree)
        XCTAssertEqual(RestormerVariant.motionDeblur.normKind, .withBias)
        XCTAssertEqual(RestormerVariant.defocusDeblur.normKind, .withBias)
    }

    /// The key-set difference is real and measurable without loading any weights: build both trees
    /// and count. 494 vs 406 — the 88 missing tensors are exactly the LayerNorm biases.
    func testBiasFreeTreeHasEightyEightFewerTensors() {
        let withBias = Restormer(Restormer.Configuration()).parameters().flattened().count
        let biasFree = Restormer(.denoising).parameters().flattened().count
        XCTAssertEqual(withBias, 494)
        XCTAssertEqual(biasFree, 406)
        XCTAssertEqual(withBias - biasFree, 88)
    }

    /// Tile geometry is 8-aligned because three `pixelUnshuffle(2)` stages make the grouping grid
    /// phase-sensitive to the tile origin. The core rounds down; pin that so it cannot regress.
    func testTileGeometryDefaultsAreEightAligned() {
        let c = RestormerConfiguration()
        XCTAssertEqual(c.tile % 8, 0, "tile default must be 8-aligned")
        XCTAssertEqual(c.overlap % 8, 0, "overlap default must be 8-aligned")
        // Measured: overlap 0 leaves a faint seam (1.31x boundary gradient); >= 8 is clean.
        XCTAssertGreaterThanOrEqual(c.overlap, 8, "overlap 0 leaves a measurable seam")
        XCTAssertGreaterThan(c.tile, 2 * c.overlap)
    }
}
