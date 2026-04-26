# English–Russian Word Alignment with Progressive Modeling

A Swift implementation of IBM Model 1 and IBM Model 2 word alignment for English–Russian translation pairs, trained on the News Commentary v16 corpus from WMT22.

## Project Structure
Sources/WordAlignment/
    main.swift          # Entry point: training and evaluation pipeline
    Corpus.swift        # Corpus loading utilities
    IBM_Model1.swift    # IBM Model 1 with EM training
    IBM_Model2.swift    # IBM Model 2 with distortion parameters
    Aligner.swift       # Viterbi alignment and symmetrization


## Data

We use the **News Commentary v16** English–Russian parallel corpus obtained from [WMT22](https://www.statmt.org/wmt22/translation-task.html), consisting of 331,508 sentence pairs.

Download the data:
```bash
mkdir -p data
cd data
curl -O https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.en-ru.tsv.gz
gunzip news-commentary-v16.en-ru.tsv.gz
```

## Models

### IBM Model 1
Learns translation probabilities t(f|e) using the EM algorithm, assuming uniform alignment probabilities.

### IBM Model 2
Extends Model 1 by adding a distortion parameter q(j|i, lenF, lenE) that models alignment position probabilities.

## Results

| Model | Perplexity (en→ru) |
|-------|-------------------|
| IBM Model 1 | 67.51 |
| IBM Model 2 | 34.96 |

Model 2 achieves significantly lower perplexity, demonstrating the benefit of modeling alignment positions in addition to translation probabilities.

## Bidirectional Alignment

Both models are trained in two directions (English→Russian and Russian→English). The alignment outputs are combined using:
- **Intersection**: only links agreed upon by both directions (high precision)
- **Union**: all links from both directions (high recall)

## Sample Alignments
Pair: "since their articles appeared, the price of gold has moved up still further."
→  "с тех пор как вышли их статьи, стоимость золота повысилась еще больше."
Model 1 intersection: 0-1 1-5 3-9 7-8 11-10
Model 2 intersection: 0-1 1-5 3-6 7-8 11-10 12-11

## How to Run

```bash
swift run
```

## Course

CS494 – Natural Language Processing  
University of Alaska Fairbanks
