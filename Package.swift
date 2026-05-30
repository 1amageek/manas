// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "manas",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(
            name: "manas",
            targets: ["manas"]
        ),
        .library(
            name: "ManasCore",
            targets: ["ManasCore"]
        ),
        .library(
            name: "ManasRuntime",
            targets: ["ManasRuntime"]
        ),
        .library(
            name: "ManasMLXModels",
            targets: ["ManasMLXModels"]
        ),
        .library(
            name: "ManasMLXTraining",
            targets: ["ManasMLXTraining"]
        ),
        .library(
            name: "ManasMLXRuntime",
            targets: ["ManasMLXRuntime"]
        ),
    ],
    dependencies: [
        .package(path: "../manas-training-data"),
        .package(path: "../embodiment-contract"),
        .package(url: "https://github.com/apple/swift-log", from: "1.13.1"),
        .package(url: "https://github.com/apple/swift-configuration", from: "1.2.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
    ],
    targets: [
        .target(
            name: "ManasCore",
            dependencies: [
                .product(name: "EmbodimentContract", package: "embodiment-contract"),
            ]
        ),
        .target(
            name: "ManasRuntime",
            dependencies: [
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Configuration", package: "swift-configuration"),
            ]
        ),
        .target(
            name: "ManasMLXModels",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ManasMLXTraining",
            dependencies: [
                .product(name: "ManasTrainingData", package: "manas-training-data"),
                "ManasMLXModels",
                "ManasCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ManasMLXRuntime",
            dependencies: [
                "ManasCore",
                "ManasMLXModels",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "manas",
            dependencies: [
                "ManasCore",
                "ManasRuntime",
            ]
        ),
        .testTarget(
            name: "manasTests",
            dependencies: [
                "ManasCore",
                "ManasRuntime",
                "ManasMLXModels",
                "ManasMLXTraining",
                "ManasMLXRuntime",
                .product(name: "ManasTrainingData", package: "manas-training-data"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
