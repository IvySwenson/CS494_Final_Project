import Foundation

// Load News Commentary v16 English-Russian parallel corpus from WMT22
let tsvPath = "/Users/xuejunchen/Documents/GitHub/CS494_Final_Project/data/news-commentary-v16.en-ru.tsv"
let (english, russian) = loadTSVCorpus(from: tsvPath)
print("Loaded \(english.count) sentence pairs")

// --- IBM Model 1 ---
print("\nTraining Model 1 English -> Russian...")
let m1enRu = IBM_Model1(source: english, target: russian, iterations: 5)
print("Training Model 1 Russian -> English...")
let m1ruEn = IBM_Model1(source: russian, target: english, iterations: 5)

let ppl1 = m1enRu.computePerplexity(source: english, target: russian)
print("Model 1 Perplexity (en->ru): \(ppl1)")

// --- IBM Model 2 ---
print("\nTraining Model 2 English -> Russian...")
let m2enRu = IBM_Model2(source: english, target: russian, iterations: 5)
print("Training Model 2 Russian -> English...")
let m2ruEn = IBM_Model2(source: russian, target: english, iterations: 5)

let ppl2 = m2enRu.computePerplexity(source: english, target: russian)
print("Model 2 Perplexity (en->ru): \(ppl2)")

// --- Sample Alignments: Model 1 vs Model 2 ---
print("\n--- Sample Alignments (Model 1 vs Model 2) ---")
for i in 0..<5 {
    let eng = english[i]
    let rus = russian[i]

    // Model 1 bidirectional alignment
    let fwd1 = viterbiAlign(source: eng, target: rus, model: m1enRu)
    let bwd1 = viterbiAlign(source: rus, target: eng, model: m1ruEn)
    let inter1 = intersect(forward: fwd1, backward: bwd1)

    // Model 2 bidirectional alignment
    let fwd2 = viterbiAlign(source: eng, target: rus, model: m2enRu)
    let bwd2 = viterbiAlign(source: rus, target: eng, model: m2ruEn)
    let inter2 = intersect(forward: fwd2, backward: bwd2)

    print("\nPair \(i+1):")
    print("  EN: \(eng.joined(separator: " "))")
    print("  RU: \(rus.joined(separator: " "))")
    print("  Model 1 intersection: \(inter1.map { "\($0.src)-\($0.tgt)" }.joined(separator: " "))")
    print("  Model 2 intersection: \(inter2.map { "\($0.src)-\($0.tgt)" }.joined(separator: " "))")
}