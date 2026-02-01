import Foundation
import Testing
@testable import manas

@Test func baselineCoverageBuildsConfig() async throws {
    let rates = try UpdateRates(controllerUpdate: 0.01, sensorSample: 0.01, actuatorUpdate: 0.01)
    let config = try ConformanceTemplate.baselineCoverage(
        updateRates: rates,
        stepTime: 0.1,
        amplitudeBands: [try Band(minimum: 0.1, maximum: 0.2)],
        slopeBands: [try Band(minimum: 0.0, maximum: 0.0)],
        frequencyBands: [try Band(minimum: 1.0, maximum: 2.0)],
        seedBase: 100,
        perturbationDelta: 0.01,
        modeInductionOffsets: [-0.01, 0.0, 0.01],
        minimumCutoffHz: 0.5
    )
    #expect(config.duration == 0.1)
}
