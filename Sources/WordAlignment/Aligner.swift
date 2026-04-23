import Foundation

public typealias AlignmentLink = (src: Int, tgt: Int)
public typealias Alignment = [AlignmentLink]

/// Viterbi alignment for Model 1
public func viterbiAlign(source: Sentence, target: Sentence, model: IBM_Model1) -> Alignment {
    var alignment: Alignment = []
    for (i, srcWord) in source.enumerated() {
        var bestJ = 0
        var bestScore = -1.0
        for (j, tgtWord) in target.enumerated() {
            let score = model.translationProb(e: tgtWord, f: srcWord)
            if score > bestScore {
                bestScore = score
                bestJ = j
            }
        }
        alignment.append((src: i, tgt: bestJ))
    }
    return alignment
}

/// Viterbi alignment for Model 2 (uses both t and q)
public func viterbiAlign(source: Sentence, target: Sentence, model: IBM_Model2) -> Alignment {
    var alignment: Alignment = []
    let lenF = source.count
    let lenE = target.count
    for (i, srcWord) in source.enumerated() {
        var bestJ = 0
        var bestScore = -1.0
        for (j, tgtWord) in target.enumerated() {
            let t = model.translationProb(e: tgtWord, f: srcWord)
            let q = model.distortionProb(j: i, i: j, lenF: lenF, lenE: lenE)
            let score = t * q
            if score > bestScore {
                bestScore = score
                bestJ = j
            }
        }
        alignment.append((src: i, tgt: bestJ))
    }
    return alignment
}

/// Intersection: only keep links that appear in both directions
public func intersect(forward: Alignment, backward: Alignment) -> Alignment {
    let bwdSet = Set(backward.map { "\($0.tgt)-\($0.src)" })
    return forward.filter { bwdSet.contains("\($0.src)-\($0.tgt)") }
}

/// Union: keep all links from both directions
public func union(forward: Alignment, backward: Alignment) -> Alignment {
    var seen = Set<String>()
    var result: Alignment = []
    for link in forward {
        let key = "\(link.src)-\(link.tgt)"
        if seen.insert(key).inserted { result.append(link) }
    }
    for link in backward {
        let key = "\(link.tgt)-\(link.src)"
        if seen.insert(key).inserted { result.append((src: link.tgt, tgt: link.src)) }
    }
    return result
}

/// Print alignment in standard i-j format
public func printAlignment(_ alignment: Alignment) {
    let str = alignment.map { "\($0.src)-\($0.tgt)" }.joined(separator: " ")
    print(str)
}