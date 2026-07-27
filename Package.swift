// swift-tools-version: 6.2
import PackageDescription

// mlx-restormer-swift — Restormer image restoration for MLXEngine.
// One architecture, several products: motion deblur · single-image defocus deblur · real denoise.
// Upstream: swz30/Restormer (plain MIT). See PORT-STATUS.md.
let package = Package(
    name: "mlx-restormer-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "RestormerMLXCore", targets: ["RestormerMLXCore"]),
        .library(name: "MLXRestormer", targets: ["MLXRestormer"]),
        .executable(name: "restormer-gate", targets: ["RestormerGate"]),
        .executable(name: "restormer-validate", targets: ["RestormerValidate"]),
    ],
    dependencies: [
        // 0.38.0 = contract 1.29.0, and the first tag carrying `licenseEnforcement`
        // (contract 1.28.0 / v0.37.0), which the validate target sets to `.blocking` to match
        // how Forge constructs the engine in production.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.38.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "RestormerMLXCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "MLXRestormer",
            dependencies: [
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                "RestormerMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "RestormerMLXTests",
            dependencies: [
                "RestormerMLXCore", "MLXRestormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            resources: [.copy("Resources/goldens")]
        ),
        // Engine-driven authoritative footprint via MLXEngineTestKit (phys_footprint),
        // as opposed to the gate's MLX-pool `--bench` which under-reads by ~2.7x.
        .executableTarget(
            name: "RestormerValidate",
            dependencies: [
                "MLXRestormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXEngineTestKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/Validate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "RestormerGate",
            dependencies: [
                "RestormerMLXCore",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Gate",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
