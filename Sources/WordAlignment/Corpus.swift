import Foundation

/// A sentence is an array of word strings
public typealias Sentence = [String]

/// A corpus is an array of sentences
public typealias Corpus = [Sentence]

/// Load a corpus from a file where each line is a sentence and words are space-separated
public func loadCorpus(from path: String) -> Corpus {
    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
        print("Error: cannot read file at \(path)")
        return []
    }
    return content
        .components(separatedBy: .newlines)
        .filter { !$0.isEmpty }
        .map { $0.lowercased().components(separatedBy: .whitespaces).filter { !$0.isEmpty } }
}

/// Load a parallel corpus from a TSV file (English\tRussian per line)
public func loadTSVCorpus(from path: String) -> (english: Corpus, russian: Corpus) {
    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
        print("Error: cannot read file at \(path)")
        return ([], [])
    }
    var english: Corpus = []
    var russian: Corpus = []

    for line in content.components(separatedBy: .newlines) {
        let cols = line.components(separatedBy: "\t")
        guard cols.count >= 2 else { continue }
        let en = cols[0].lowercased().components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        let ru = cols[1].lowercased().components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        if en.isEmpty || ru.isEmpty { continue }
        english.append(en)
        russian.append(ru)
    }
    return (english, russian)
}