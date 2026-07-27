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
        .executable(name: "restormer-gate", targets: ["RestormerGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.0"),
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
