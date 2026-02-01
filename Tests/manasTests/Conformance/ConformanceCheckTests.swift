import Testing
@testable import manas

@Test func outputSnappingDetectsSmallClusterCount() async throws {
    let series = [
        [0.0, 1.0],
        [0.01, 0.99],
        [0.0, 1.02],
    ]
    let result = OutputSnappingCheck.evaluate(series: series, epsilon: 0.05, maxClusters: 2)
    #expect(result.passes == false)
}

@Test func phaseVarianceFailsWhenTooLow() async throws {
    let phases: [PhaseIndex: [Double]] = [
        PhaseIndex(0): [1.0, 1.0, 1.0],
    ]
    let result = PhaseVarianceCheck.evaluate(phases: phases, minimumVariance: 0.01)
    #expect(result.passes == false)
}

@Test func phaseSnappingDetectsClustering() async throws {
    let phases: [PhaseIndex: [Double]] = [
        PhaseIndex(0): [0.0, 0.01, 0.0, 0.02],
    ]
    let result = PhaseSnappingCheck.evaluate(phases: phases, epsilon: 0.05, maxClusters: 2)
    #expect(result.passes == false)
}

@Test func modeInductionDetectsLowModeCount() async throws {
    let steadyStates = [
        [0.0, 0.0],
        [0.01, 0.0],
        [0.02, 0.0],
    ]
    let result = ModeInductionCheck.evaluate(steadyStates: steadyStates, epsilon: 0.05, maxModes: 2)
    #expect(result.passes == false)
}
