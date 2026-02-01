public extension ConformancePlan {
    var allFamilies: [InputFamily] {
        var families: [InputFamily] = []
        families.reserveCapacity(
            continuityPairs.count * 2
                + totalVariationFamilies.count
                + snappingFamilies.count
                + phaseFamilies.count
                + modeInductionFamilies.reduce(0) { $0 + $1.count }
        )

        for pair in continuityPairs {
            families.append(pair.base)
            families.append(pair.perturbed)
        }
        families.append(contentsOf: totalVariationFamilies)
        families.append(contentsOf: snappingFamilies)
        families.append(contentsOf: phaseFamilies)
        for group in modeInductionFamilies {
            families.append(contentsOf: group)
        }
        return families
    }
}

