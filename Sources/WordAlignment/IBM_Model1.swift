import Foundation

public struct IBM_Model1 {

    public struct Vocab {
        private(set) var toId: [String: Int] = [:]
        private(set) var toStr: [String] = []

        public init() {}

        @discardableResult
        public mutating func id(_ s: String) -> Int {
            if let i = toId[s] { return i }
            let i = toStr.count
            toStr.append(s)
            toId[s] = i
            return i
        }

        public func lookup(_ s: String) -> Int? { toId[s] }
        public func str(_ i: Int) -> String { toStr[i] }
        public var size: Int { toStr.count }
    }

    private var eVocab = Vocab()
    private var fVocab = Vocab()
    private var tTable: [Int: [Int: Double]] = [:]
    private var cooccur: [Int: Set<Int>] = [:]
    private let floorProb: Double = 1e-12

    public init(source: Corpus, target: Corpus, iterations: Int = 5) {
        buildCooccurrence(source: source, target: target)
        initializeUniform()
        if iterations > 0 {
            trainEM(source: source, target: target, iterations: iterations)
        }
    }

    public func translationProb(e: String, f: String) -> Double {
        guard
            let eId = eVocab.lookup(e),
            let fId = fVocab.lookup(f),
            let row = tTable[eId],
            let p = row[fId]
        else { return floorProb }
        return p
    }

    private mutating func buildCooccurrence(source: Corpus, target: Corpus) {
        cooccur.removeAll(keepingCapacity: true)
        eVocab = Vocab()
        fVocab = Vocab()

        for (fSent, eSent) in zip(source, target) {
            let fIds = Set(fSent.map { fVocab.id($0) })
            let eIds = Set(eSent.map { eVocab.id($0) })
            for eId in eIds {
                cooccur[eId, default: Set<Int>()].formUnion(fIds)
            }
        }
    }

    private mutating func initializeUniform() {
        tTable.removeAll(keepingCapacity: true)
        for (eId, fset) in cooccur {
            let p = 1.0 / Double(max(fset.count, 1))
            var row: [Int: Double] = [:]
            row.reserveCapacity(fset.count)
            for fId in fset { row[fId] = p }
            tTable[eId] = row
        }
    }

    private mutating func trainEM(source: Corpus, target: Corpus, iterations: Int) {
        var prevPPL = Double.infinity

        for _ in 0..<iterations {
            var count: [Int: [Int: Double]] = [:]
            var total: [Int: Double] = [:]

            for (fSent, eSent) in zip(source, target) {
                let eIds: [Int] = eSent.compactMap { eVocab.lookup($0) }
                if eIds.isEmpty { continue }

                for fTok in fSent {
                    guard let fId = fVocab.lookup(fTok) else { continue }

                    var denom = 0.0
                    for eId in eIds {
                        denom += (tTable[eId]?[fId] ?? floorProb)
                    }
                    if denom <= 0 { continue }

                    for eId in eIds {
                        let t = (tTable[eId]?[fId] ?? floorProb)
                        let frac = t / denom
                        if frac <= 0 { continue }
                        count[eId, default: [:]][fId, default: 0.0] += frac
                        total[eId, default: 0.0] += frac
                    }
                }
            }

            for (eId, fset) in cooccur {
                let denom = total[eId] ?? 0.0
                if denom <= 0 { continue }
                var newRow: [Int: Double] = [:]
                newRow.reserveCapacity(fset.count)
                let countedRow = count[eId] ?? [:]
                for fId in fset {
                    newRow[fId] = max((countedRow[fId] ?? 0.0) / denom, floorProb)
                }
                tTable[eId] = newRow
            }

            let ppl = computePerplexity(source: source, target: target)
            if prevPPL.isFinite {
                let relImprove = (prevPPL - ppl) / max(prevPPL, 1e-12)
                if relImprove >= 0 && relImprove < 1e-4 { break }
            }
            prevPPL = ppl
        }
    }

    public func computePerplexity(source: Corpus, target: Corpus) -> Double {
        var totalLogProb = 0.0
        var tokenCount = 0

        for (fSent, eSent) in zip(source, target) {
            let eIds: [Int] = eSent.compactMap { eVocab.lookup($0) }
            let eLen = max(eIds.count, 1)
            let invELen = 1.0 / Double(eLen)

            for fTok in fSent {
                guard let fId = fVocab.lookup(fTok) else { continue }
                var sumT = 0.0
                for eId in eIds {
                    sumT += (tTable[eId]?[fId] ?? floorProb)
                }
                let p = max(sumT * invELen, floorProb)
                totalLogProb += log(p)
                tokenCount += 1
            }
        }

        if tokenCount == 0 { return Double.infinity }
        return exp(-totalLogProb / Double(tokenCount))
    }
}