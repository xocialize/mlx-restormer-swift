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
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.36.0"),
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
