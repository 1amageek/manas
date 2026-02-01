// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "manas",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "manas",
            targets: ["manas"]
        ),
        .library(
            name: "ManasMLX",
            targets: ["ManasMLX"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-log", from: "1.9.1"),
        .package(url: "https://github.com/apple/swift-configuration", from: "1.0.2"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "manas",
            dependencies: [
                .product(name: "Logging", package: "swift-log"),
                .product(name: "Configuration", package: "swift-configuration"),
            ]
        ),
        .target(
            name: "ManasMLX",
            dependencies: [
                "manas",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "manasTests",
            dependencies: [
                "manas",
                "ManasMLX",
            ]
        ),
    ]
)
