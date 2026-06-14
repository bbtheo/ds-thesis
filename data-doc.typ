// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): set block(
    fill: luma(230),
    width: 100%,
    inset: 8pt,
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.abs
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == str {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == content {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subrefnumbering: "1a",
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => numbering(subrefnumbering, n-super, quartosubfloatcounter.get().first() + 1))
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => {
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          }

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != str {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    block(below: 0pt, new_title_block) +
    old_callout.body.children.at(1))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#fa-info()" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black, body_background_color: white) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: body_background_color, width: 100%, inset: 8pt, body))
      }
    )
}



#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  lang: "en",
  region: "US",
  font: "libertinus serif",
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: "libertinus serif",
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  sectionnumbering: none,
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)
  if title != none {
    align(center)[#block(inset: 2em)[
      #set par(leading: heading-line-height)
      #if (heading-family != none or heading-weight != "bold" or heading-style != "normal"
           or heading-color != black) {
        set text(font: heading-family, weight: heading-weight, style: heading-style, fill: heading-color)
        text(size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(size: subtitle-size)[#subtitle]
        }
      } else {
        text(weight: "bold", size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(weight: "bold", size: subtitle-size)[#subtitle]
        }
      }
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth,
      indent: toc_indent
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: none
)

#set page(
  paper: "us-letter",
  margin: (x: 1.25in, y: 1.25in),
  numbering: "1",
)

#show: doc => article(
  title: [Fraud Detection Datasets],
  subtitle: [Data sources, schema overview, and exploratory analysis],
  date: [2026-04-04],
  toc: true,
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)

= Overview
<overview>
Six publicly available datasets were ingested into a local DuckDB database for this thesis project. All datasets contain labelled transaction records suitable for supervised fraud detection. The table below summarises key properties.

#table(
  columns: 3,
  align: (left,right,right,),
  table.header([Dataset], [Rows], [Columns],),
  table.hline(),
  [EU Credit Card (MLG-ULB)], [284,807], [31],
  [Credit Card Fraud 2025], [500,000], [16],
  [AI Banking 2025], [10,000], [11],
  [BankSim (transactions)], [594,643], [10],
  [BankSim (network)], [594,643], [5],
  [PaySim], [6,362,620], [11],
  [FiFAR (train)], [506,118], [37],
  [FiFAR (test)], [96,843], [34],
)

#horizontalrule

= European Credit Card Fraud (MLG-ULB)
<sec-eu-cc>
The canonical benchmark dataset released by the Machine Learning Group at Université Libre de Bruxelles @Dal2015Calibrating. It contains two days of credit card transactions by European cardholders in September 2013. Features V1--V28 are PCA-transformed to protect cardholder identity; only `Time` (seconds since the first transaction) and `Amount` are in original scale. The label `Class` is 1 for fraud.

#table(
  columns: 2,
  align: (left,left,),
  table.header([Column], [Type],),
  table.hline(),
  [Time], [BIGINT],
  [V1], [DOUBLE],
  [V2], [DOUBLE],
  [V3], [DOUBLE],
  [V4], [DOUBLE],
  [V5], [DOUBLE],
  [V6], [DOUBLE],
  [V7], [DOUBLE],
  [V8], [DOUBLE],
  [V9], [DOUBLE],
  [V10], [DOUBLE],
  [V11], [DOUBLE],
  [V12], [DOUBLE],
  [V13], [DOUBLE],
  [V14], [DOUBLE],
  [V15], [DOUBLE],
  [V16], [DOUBLE],
  [V17], [DOUBLE],
  [V18], [DOUBLE],
  [V19], [DOUBLE],
  [V20], [DOUBLE],
  [V21], [DOUBLE],
  [V22], [DOUBLE],
  [V23], [DOUBLE],
  [V24], [DOUBLE],
  [V25], [DOUBLE],
  [V26], [DOUBLE],
  [V27], [DOUBLE],
  [V28], [DOUBLE],
  [Amount], [DOUBLE],
  [Class], [BIGINT],
)
#block[
#block[
```
Total transactions: 284,807  |  Fraud: 492 (0.17%)  |  Amount: min $0, mean $88.35, max $25691.16  |  Duration: 48 hours
```

]
]
#box(image("data-doc_files/figure-typst/eu-cc-plots-1.svg"))

#box(image("data-doc_files/figure-typst/eu-cc-time-1.svg"))

#horizontalrule

= Credit Card Fraud 2025 (Synthetic)
<sec-cc-2025>
A synthetic credit card fraud dataset generated in 2025 with realistic multi-country transaction patterns. Contains 500,000 transactions with 16 features.

#table(
  columns: 2,
  align: (left,left,),
  table.header([Column], [Type],),
  table.hline(),
  [Transaction\_ID], [BIGINT],
  [Customer\_ID], [BIGINT],
  [Transaction\_Date], [TIMESTAMP],
  [Amount], [DOUBLE],
  [Merchant\_Category], [VARCHAR],
  [Merchant\_ID], [BIGINT],
  [Card\_Type], [VARCHAR],
  [Transaction\_Type], [VARCHAR],
  [Country], [VARCHAR],
  [Is\_International], [BIGINT],
  [Is\_Chip], [BIGINT],
  [Is\_Pin\_Used], [BIGINT],
  [Distance\_From\_Home], [DOUBLE],
  [Hour\_of\_Day], [BIGINT],
  [Device\_Type], [VARCHAR],
  [Fraud\_Flag], [BIGINT],
)
#block[
#block[
```
Total transactions: 500,000  |  Fraud: 7,500 (1.50%)  |  Label column: fraud_flag
```

]
]
#box(image("data-doc_files/figure-typst/cc25-amounts-1.svg"))

#horizontalrule

= AI Banking Fraud 2025 (Synthetic)
<sec-ai-banking>
A small synthetic dataset (10,000 transactions) designed for educational and small-scale experiments. Features include account age, credit score, transaction type and previous fraud history.

#table(
  columns: 2,
  align: (left,left,),
  table.header([Column], [Type],),
  table.hline(),
  [Transaction\_ID], [VARCHAR],
  [Customer\_ID], [VARCHAR],
  [Transaction\_Amount], [DOUBLE],
  [Transaction\_Type], [VARCHAR],
  [Transaction\_Location], [VARCHAR],
  [Transaction\_Time], [TIMESTAMP],
  [Device\_Used], [VARCHAR],
  [Account\_Age], [BIGINT],
  [Credit\_Score], [BIGINT],
  [Previous\_Fraud], [BIGINT],
  [Is\_Fraud], [BIGINT],
)
#block[
#block[
```
Total: 10,000  |  Fraud: 2,838 (28.38%)  |  Amount: min $10.89, mean $4997.49, max $9999.29
```

]
]
#box(image("data-doc_files/figure-typst/ai-plots-1.svg"))

#horizontalrule

= BankSim
<sec-banksim>
BankSim is an agent-based payment simulator calibrated from aggregated data of a Spanish bank @Lopez2014BankSim. It generates labelled transactions over a six-month window. Two files are available: the main transaction log and a network (pairwise) summary.

== Transaction log
<transaction-log>
#table(
  columns: 2,
  align: (left,left,),
  table.header([Column], [Type],),
  table.hline(),
  [step], [BIGINT],
  [customer], [VARCHAR],
  [age], [VARCHAR],
  [gender], [VARCHAR],
  [zipcodeOri], [BIGINT],
  [merchant], [VARCHAR],
  [zipMerchant], [BIGINT],
  [category], [VARCHAR],
  [amount], [DOUBLE],
  [fraud], [BIGINT],
)
#block[
#block[
```
Transactions: 594,643  |  Fraud: 7,200 (1.21%)  |  Mean amount: $37.89
Customers: 4,112  |  Merchants: 50  |  Categories: 15
```

]
]
#box(image("data-doc_files/figure-typst/banksim-plot-1.svg"))

== Network table
<network-table>
#table(
  columns: 2,
  align: (left,left,),
  table.header([Column], [Type],),
  table.hline(),
  [Source], [VARCHAR],
  [Target], [VARCHAR],
  [Weight], [DOUBLE],
  [typeTrans], [VARCHAR],
  [fraud], [BIGINT],
)
#block[
#block[
```
Network edges (customer–merchant pairs): 594,643
```

]
]

#horizontalrule

= PaySim
<sec-paysim>
PaySim is a mobile money simulator based on one month of anonymised transaction logs from a real mobile money service operating in an African country @Lopez2016PaySim. It is widely used for fraud detection research due to its scale and realistic transfer patterns. Fraud occurs exclusively in `TRANSFER` and `CASH_OUT` transaction types.

#table(
  columns: 2,
  align: (left,left,),
  table.header([Column], [Type],),
  table.hline(),
  [step], [BIGINT],
  [type], [VARCHAR],
  [amount], [DOUBLE],
  [nameOrig], [VARCHAR],
  [oldbalanceOrg], [DOUBLE],
  [newbalanceOrig], [DOUBLE],
  [nameDest], [VARCHAR],
  [oldbalanceDest], [DOUBLE],
  [newbalanceDest], [DOUBLE],
  [isFraud], [BIGINT],
  [isFlaggedFraud], [BIGINT],
)
#block[
#block[
```
Transactions: 6,362,620  |  Fraud: 8,213 (0.13%)  |  Mean amount: $179861.9  |  Steps (hours): 743
```

]
]
#box(image("data-doc_files/figure-typst/paysim-by-type-1.svg"))

#box(image("data-doc_files/figure-typst/paysim-time-1.svg"))

#horizontalrule

= FiFAR --- Financial Fraud Alert Review Dataset
<sec-fifar>
FiFAR is a synthetic dataset released by Feedzai representing bank account opening applications reviewed by fraud analysts @Vaz2023FiFAR. It accompanies research on human-AI collaboration and learning-to-defer systems. The dataset contains one million applications split across train and test sets, with predictions from 50 synthetic fraud analysts. For this thesis the main focus is on the transaction-level features and the binary fraud label.

#table(
  columns: 2,
  align: (left,left,),
  table.header([Column], [Type],),
  table.hline(),
  [case\_id], [BIGINT],
  [fraud\_bool], [BIGINT],
  [income], [DOUBLE],
  [name\_email\_similarity], [DOUBLE],
  [prev\_address\_months\_count], [BIGINT],
  [current\_address\_months\_count], [BIGINT],
  [customer\_age], [BIGINT],
  [days\_since\_request], [DOUBLE],
  [intended\_balcon\_amount], [DOUBLE],
  [payment\_type], [VARCHAR],
  [zip\_count\_4w], [BIGINT],
  [velocity\_6h], [DOUBLE],
  [velocity\_24h], [DOUBLE],
  [velocity\_4w], [DOUBLE],
  [bank\_branch\_count\_8w], [BIGINT],
  [date\_of\_birth\_distinct\_emails\_4w], [BIGINT],
  [employment\_status], [VARCHAR],
  [credit\_risk\_score], [BIGINT],
  [email\_is\_free], [BIGINT],
  [housing\_status], [VARCHAR],
  [phone\_home\_valid], [BIGINT],
  [phone\_mobile\_valid], [BIGINT],
  [bank\_months\_count], [BIGINT],
  [has\_other\_cards], [BIGINT],
  [proposed\_credit\_limit], [DOUBLE],
  [foreign\_request], [BIGINT],
  [source], [VARCHAR],
  [session\_length\_in\_minutes], [DOUBLE],
  [device\_os], [VARCHAR],
  [keep\_alive\_session], [BIGINT],
  [device\_distinct\_emails\_8w], [BIGINT],
  [device\_fraud\_count], [BIGINT],
  [month], [BIGINT],
  [model\_score], [DOUBLE],
  [batch], [BIGINT],
  [assignment], [VARCHAR],
  [decision], [DOUBLE],
)
#block[
#block[
```
Train rows: 506,118  |  Test rows: 96,843  |  Fraud in train: 5,705 (1.13%)  |  Label column: fraud_bool
```

]
]
#box(image("data-doc_files/figure-typst/fifar-numeric-dist-1.svg"))

#horizontalrule

= Class Imbalance Comparison
<sec-imbalance>
All datasets exhibit significant class imbalance, a defining challenge in fraud detection.

#box(image("data-doc_files/figure-typst/imbalance-plot-1.svg"))

#horizontalrule
