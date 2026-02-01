import Testing
@testable import manas

@Test func phaseBandwidthRejectsFastSignal() async throws {
    let phases: [PhaseIndex: [Double]] = [
        PhaseIndex(0): [0.0, 1.0, 0.0],
    ]
    let result = try PhaseBandwidthCheck.evaluate(
        phases: phases,
        deltaTime: 0.01,
        bandwidthHz: 1.0
    )
    #expect(result.passes == false)
}

