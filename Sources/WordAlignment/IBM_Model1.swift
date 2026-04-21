import Foundation

/// IBM Model 1 word aligner using Expectation-Maximization (EM) algorithm.
/// Learns translation probabilities t(f|e) from a parallel corpus.
public struct IBM_Model1 {

    /// Bidirectional vocabulary mapping between word strings and integer IDs.
    /// Using integer IDs instead of strings speeds up table lookups during training.
    public struct Vocab {
        private(set) var toId: [String: Int] = [:]
        private(set) var toStr: [String] = []

        public init() {}

        /// Add a word to the vocabulary and return its ID.
        /// If the word already exists, return the existing ID.
        @discardableResult
        public mutating func id(_ s: String) -> Int {
            if let i = toId[s] { return i }
            let i = toStr.count
            toStr.append(s)
            toId[s] = i
            return i
        }

        /// Look up a word's ID without adding it to the vocabulary.
        public func lookup(_ s: String) -> Int? { toId[s] }
        public func str(_ i: Int) -> String { toStr[i] }
        public var size: Int { toStr.count }
    }

    private var eVocab = Vocab()   // English vocabulary
    private var fVocab = Vocab()   // Foreign (Russian) vocabulary

    /// tTable[eId][fId] = t(f|e) = P(foreign word f | english word e)
    private var tTable: [Int: [Int: Double]] = [:]

    /// cooccur[eId] = set of foreign word IDs that co-occur with english word eId
    /// Used to restrict which parameters exist, keeping the table sparse.
    private var cooccur: [Int: Set<Int>] = [:]

    /// Smoothing floor to avoid exact zero probabilities
    private let floorProb: Double = 1e-12

    /// Initialize and train IBM Model 1 on a parallel corpus.
    /// - Parameters:
    ///   - source: Foreign (Russian) corpus
    ///   - target: English corpus
    ///   - iterations: Maximum number of EM iterations
    public init(source: Corpus, target: Corpus, iterations: Int = 5) {
        buildCooccurrence(source: source, target: target)
        initializeUniform()
        if iterations > 0 {
            trainEM(source: source, target: target, iterations: iterations)
        }
    }

    /// Return the learned translation probability t(f|e).
    public func translationProb(e: String, f: String) -> Double {
        guard
            let eId = eVocab.lookup(e),
            let fId = fVocab.lookup(f),
            let row = tTable[eId],
            let p = row[fId]
        else { return floorProb }
        return p
    }

    // MARK: - Training helpers

    /// Scan the parallel corpus to build vocabularies and co-occurrence sets.
    /// Co-occurrence sets define which t(f|e) parameters actually exist.
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

    /// Initialize t(f|e) uniformly over all foreign words co-occurring with each english word.
    /// Each english word distributes probability mass equally across its co-occurring foreign words.
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

    /// Run the EM algorithm to estimate translation probabilities.
    /// E-step: compute expected alignment counts using current parameters.
    /// M-step: re-estimate parameters from expected counts.
    /// Stops early if perplexity improvement is below threshold.
    private mutating func trainEM(source: Corpus, target: Corpus, iterations: Int) {
        var prevPPL = Double.infinity

        for _ in 0..<iterations {

            // E-step: accumulate fractional counts
            var count: [Int: [Int: Double]] = [:]
            var total: [Int: Double] = [:]

            for (fSent, eSent) in zip(source, target) {
                let eIds: [Int] = eSent.compactMap { eVocab.lookup($0) }
                if eIds.isEmpty { continue }

                for fTok in fSent {
                    guard let fId = fVocab.lookup(fTok) else { continue }

                    // Compute normalization: sum of t(f|e) over all english words in sentence
                    var denom = 0.0
                    for eId in eIds {
                        denom += (tTable[eId]?[fId] ?? floorProb)
                    }
                    if denom <= 0 { continue }

                    // Distribute fractional count proportionally
                    for eId in eIds {
                        let t = (tTable[eId]?[fId] ?? floorProb)
                        let frac = t / denom
                        if frac <= 0 { continue }
                        count[eId, default: [:]][fId, default: 0.0] += frac
                        total[eId, default: 0.0] += frac
                    }
                }
            }

            // M-step: normalize counts to get new probabilities
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

            // Check convergence: stop if relative improvement is tiny
            let ppl = computePerplexity(source: source, target: target)
            if prevPPL.isFinite {
                let relImprove = (prevPPL - ppl) / max(prevPPL, 1e-12)
                if relImprove >= 0 && relImprove < 1e-4 { break }
            }
            prevPPL = ppl
        }
    }

    /// Compute perplexity of the model on a parallel corpus.
    /// Lower perplexity indicates a better model fit.
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
                // p(f | eSent) = (1/|eSent|) * sum_e t(f|e)
                let p = max(sumT * invELen, floorProb)
                totalLogProb += log(p)
                tokenCount += 1
            }
        }

        if tokenCount == 0 { return Double.infinity }
        return exp(-totalLogProb / Double(tokenCount))
    }
}